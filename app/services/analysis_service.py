import asyncio
import faiss
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi.concurrency import run_in_threadpool
from app.models.document import Article, ArticleRecommend, ArticleRecommendVector, UserRecentArticle
import pickle

class AnalysisService:
    def __init__(self):
        """
        초기화 시 모델 로딩은 하지 않고, 인덱스만 준비
        실제 모델 로딩은 load_and_build_index()에서 수행
        """
        print("AnalysisService 초기화 중...")
        self.model = None
        self.d = 768
        self.index = faiss.IndexFlatIP(self.d)
        self.index_to_reco_id = {}
        self.index_lock = asyncio.Lock()
        self.vector_id_to_article_id = {}

    async def _ensure_model_loaded(self):
        """모델이 로드되지 않았다면 비동기로 로드"""
        if self.model is None:
            print("SBERT 모델을 로드합니다...")
            def _load():
                return SentenceTransformer('jhgan/ko-sroberta-multitask')
            self.model = await run_in_threadpool(_load)

    async def encode_text(self, text: str) -> np.ndarray:
        """SBERT 임베딩을 비동기 실행 (스레드풀 사용)"""
        await self._ensure_model_loaded()
        embedding = await run_in_threadpool(self.model.encode, text)
        return np.asarray(embedding, dtype='float32')

    async def load_and_build_index(self, db: AsyncSession):
        """
        서버 시작 시 한 번 실행:
        - 모든 기사 벡터를 DB에서 불러와 FAISS 인덱스를 빌드
        - index_to_reco_id, vector_id_to_article_id 매핑 생성
        """
        print("DB로부터 Faiss 인덱스를 빌드합니다...")
        await self._ensure_model_loaded()

        query = (
            select(
                ArticleRecommendVector.article_recommend_id,
                ArticleRecommendVector.sbert_vector
            )
            .join(ArticleRecommend, ArticleRecommendVector.article_recommend_id == ArticleRecommend.id)
            .join(Article, Article.article_recommend_id == ArticleRecommend.id)
        )
        result = await db.execute(query)
        rows = result.all()

        if not rows:
            print("로드할 벡터가 없습니다.")
            return

        import json

        all_vectors = []
        reco_ids = []
        for reco_id, vector_data in rows:
            if isinstance(vector_data, str):
                try:
                    vector_list = json.loads(vector_data)
                except json.JSONDecodeError:
                    print(f"[WARN] JSON 파싱 실패: reco_id={reco_id}")
                    continue
            else:
                vector_list = vector_data

            if isinstance(vector_list, (list, tuple)) and len(vector_list) == self.d:
                all_vectors.append(vector_list)
                reco_ids.append(reco_id)
            else:
                print(f"[WARN] 잘못된 벡터 차원: reco_id={reco_id}, len={len(vector_list) if vector_list else None}")

        if not all_vectors:
            print("유효한 벡터가 없습니다. 인덱스 빌드 중단.")
            return

        vectors_np = np.array(all_vectors, dtype='float32')
        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, vectors_np)

            start_idx = self.index.ntotal
            await run_in_threadpool(self.index.add, vectors_np)

            for i, reco_id in enumerate(reco_ids):
                self.index_to_reco_id[start_idx + i] = reco_id

            article_ids_query = (
                select(Article.id, Article.article_recommend_id)
                .where(Article.article_recommend_id.in_(reco_ids))
            )
            article_result = await db.execute(article_ids_query)
            article_rows = article_result.all()
            reco_to_article = {reco_id: article_id for article_id, reco_id in article_rows}

            for i, reco_id in enumerate(reco_ids):
                article_id = reco_to_article.get(reco_id)
                if article_id:
                    self.vector_id_to_article_id[start_idx + i] = article_id

        print(f"총 {self.index.ntotal}개의 벡터가 Faiss 인덱스에 로드되었습니다.")
        print(f"vector_id_to_article_id 매핑 수: {len(self.vector_id_to_article_id)}")

    async def add_document_to_index(self, reco_id: int, vector_list: list):
        """새로운 기사 추가 시 인덱스 업데이트"""
        vector_np = np.array([vector_list], dtype='float32')

        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, vector_np)
            start_idx = self.index.ntotal
            await run_in_threadpool(self.index.add, vector_np)
            for i in range(vector_np.shape[0]):
                self.index_to_reco_id[start_idx + i] = reco_id
        print(f"ArticleRecommend ID {reco_id}가 인덱스 {start_idx}에 추가됨")

    async def find_similar_documents_by_user(
        self, db: AsyncSession, user_id: int, top_k: int = 5
    ) -> list[int]:
        """
        [개인화 추천]
        사용자가 읽은 여러 기사들의 SBERT 벡터 평균을 계산하고,
        그 평균 벡터를 기준으로 Faiss에서 유사 기사 검색
        + 3일 이내 기사만 필터링
        + 유사도 점수 기준 내림차순 정렬
        """
        print(f"\n=== [DEBUG] find_similar_documents_by_user(user_id={user_id}) ===")

        # 1️⃣ 사용자 읽은 기사들의 추천 벡터 조회
        query = (
            select(ArticleRecommendVector.sbert_vector)
            .join(ArticleRecommend, ArticleRecommendVector.article_recommend_id == ArticleRecommend.id)
            .join(Article, Article.article_recommend_id == ArticleRecommend.id)
            .join(UserRecentArticle, UserRecentArticle.article_id == Article.id)
            .where(UserRecentArticle.user_id == user_id)
        )
        result = await db.execute(query)
        user_vectors = result.scalars().all()

        import json

        parsed_vectors = []
        for v in user_vectors:
            if isinstance(v, str):
                try:
                    parsed_vectors.append(json.loads(v))
                except json.JSONDecodeError:
                    print(f"[WARN] 잘못된 JSON 벡터 형식: {v[:80]}")
                    continue
            elif isinstance(v, (list, tuple)):
                parsed_vectors.append(v)
            else:
                print(f"[WARN] 예상치 못한 벡터 타입: {type(v)}")
        user_vectors = parsed_vectors

        if user_vectors:
            print(f"[DEBUG] 불러온 벡터 예시 타입={type(user_vectors[0])}, 길이={len(user_vectors[0])}")
        else:
            print("[DEBUG] 파싱 후 벡터 없음")

        if not user_vectors:
            print(f"[DEBUG] 사용자 {user_id}의 읽은 기사 벡터가 없습니다.")
            return []

        # 2️⃣ numpy 배열로 변환 + 차원 확인
        try:
            user_arr = np.array(user_vectors, dtype='float32')
            if user_arr.ndim == 1:
                user_arr = user_arr.reshape(1, -1)
            elif user_arr.ndim != 2:
                raise ValueError(f"벡터 차원 이상: ndim={user_arr.ndim}, 예시={user_arr[:3]}")

            if user_arr.shape[1] != self.index.d:
                print(f"[경고] 잘못된 벡터 차원 발견 ({user_arr.shape[1]} != {self.index.d}) → 필터링")
                user_arr = np.array([v for v in user_vectors if len(v) == self.index.d], dtype='float32')
                if user_arr.size == 0:
                    print("[ERROR] 유효한 벡터가 없습니다.")
                    return []
        except Exception as e:
            print(f"[ERROR] numpy 변환 실패: {repr(e)}")
            return []

        # 3️⃣ 평균 벡터 계산
        user_profile = np.mean(user_arr, axis=0).reshape(1, -1)
        print(f"[DEBUG] user_profile shape={user_profile.shape}")

        print(f"[DEBUG] index.ntotal={self.index.ntotal}, index.d={getattr(self.index, 'd', None)}")

        if getattr(self.index, "d", None) is None:
            print("[ERROR] 인덱스가 초기화되지 않았습니다. load_and_build_index() 실행 필요.")
            return []

        if user_profile.shape[1] != self.index.d:
            raise RuntimeError(
                f"[차원 불일치] user_profile={user_profile.shape[1]} / index.d={self.index.d} "
                f"→ DB 벡터 또는 모델 임베딩 차원 불일치. 인덱스 재생성 필요."
            )

        # 4️⃣ 검색 (점수와 함께 반환)
        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, user_profile)
            num_search = top_k * 3  # 여유있게 검색
            D, I = await run_in_threadpool(self.index.search, user_profile, num_search)

        # 5️⃣ 점수와 함께 reco_id 매핑 (내림차순)
        scored_reco_ids = []
        for score, faiss_index_id in zip(D[0], I[0]):
            reco_id = self.index_to_reco_id.get(faiss_index_id)
            if reco_id:
                scored_reco_ids.append((score, reco_id))
        
        # 점수 기준 내림차순 정렬
        scored_reco_ids.sort(reverse=True, key=lambda x: x[0])
        print(f"[DEBUG] 상위 5개 유사도 점수: {[f'{s:.4f}' for s, _ in scored_reco_ids[:5]]}")

        # 6️⃣ 이미 읽은 기사 제외
        read_reco_ids_query = (
            select(Article.article_recommend_id)
            .join(UserRecentArticle, UserRecentArticle.article_id == Article.id)
            .where(UserRecentArticle.user_id == user_id)
        )
        read_reco_ids_result = await db.execute(read_reco_ids_query)
        read_reco_ids = set(read_reco_ids_result.scalars().all())

        filtered_scored = [(score, rid) for score, rid in scored_reco_ids if rid not in read_reco_ids]
        
        if not filtered_scored:
            print(f"[DEBUG] 추천 가능한 새 기사 없음. (모두 이미 읽음)")
            return []

        # 7️⃣ 3일 이내 기사만 필터링 + 정렬된 순서로 반환
        three_days_ago = datetime.now() - timedelta(days=3)
        filtered_reco_ids = [rid for _, rid in filtered_scored]
        
        query_similar_articles = (
            select(Article.id, Article.article_recommend_id, Article.created_at)
            .where(
                Article.article_recommend_id.in_(filtered_reco_ids),
                Article.created_at >= three_days_ago
            )
        )
        result = await db.execute(query_similar_articles)
        recent_articles = result.all()
        
        # reco_id 순서 유지하면서 article_id 매핑
        reco_to_article = {reco_id: article_id for article_id, reco_id, _ in recent_articles}
        recommended_article_ids = [
            reco_to_article[rid] 
            for _, rid in filtered_scored 
            if rid in reco_to_article
        ][:top_k]

        print(f"[DEBUG] 추천된 기사 ID 목록 (유사도 내림차순): {recommended_article_ids}\n")

        return recommended_article_ids

    async def find_similar_documents_by_article(
        self, db: AsyncSession, article_id: int, top_k: int = 5
    ) -> list[int]:
        """
        주어진 기사(article_id)와 유사한 기사들을 Faiss 인덱스를 이용해 찾음
        + 3일 이내 기사만 필터링
        + 유사도 점수 기준 내림차순 정렬
        """
        # 1️⃣ 기준 기사 벡터 가져오기
        query = (
            select(ArticleRecommendVector.sbert_vector)
            .join(ArticleRecommend, ArticleRecommendVector.article_recommend_id == ArticleRecommend.id)
            .join(Article, Article.article_recommend_id == ArticleRecommend.id)
            .where(Article.id == article_id)
        )
        result = await db.execute(query)
        vector_row = result.scalar_one_or_none()

        if not vector_row:
            raise ValueError(f"Article ID {article_id}의 벡터를 찾을 수 없습니다.")

        if isinstance(vector_row, (bytes, bytearray)):
            query_vector = np.array(pickle.loads(vector_row)).astype("float32")
        else:
            query_vector = np.array(vector_row).astype("float32")

        # 2️⃣ FAISS 인덱스에서 유사 벡터 검색 (점수 포함)
        async with self.index_lock:
            query_vector_normalized = query_vector.reshape(1, -1)
            await run_in_threadpool(faiss.normalize_L2, query_vector_normalized)
            D, I = await run_in_threadpool(self.index.search, query_vector_normalized, top_k * 2)

        # 3️⃣ 점수와 함께 article_id 매핑 (자기 자신 제외)
        scored_articles = []
        for score, idx in zip(D[0], I[0]):
            if idx in self.vector_id_to_article_id:
                mapped_article_id = self.vector_id_to_article_id[idx]
                if mapped_article_id != article_id:  # 자기 자신 제외
                    scored_articles.append((score, mapped_article_id))

        # 4️⃣ 3일 이내 기사만 필터링
        three_days_ago = datetime.now() - timedelta(days=3)
        article_ids = [aid for _, aid in scored_articles]
        
        query_recent = (
            select(Article.id, Article.created_at)
            .where(
                Article.id.in_(article_ids),
                Article.created_at >= three_days_ago
            )
        )
        result = await db.execute(query_recent)
        recent_article_ids = {aid for aid, _ in result.all()}

        # 5️⃣ 점수 순서 유지하면서 필터링된 결과 반환
        similar_article_ids = [
            aid for score, aid in scored_articles 
            if aid in recent_article_ids
        ][:top_k]

        return similar_article_ids

# 싱글톤 인스턴스
analysis_service = AnalysisService()