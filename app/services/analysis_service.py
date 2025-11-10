import asyncio
import faiss
import numpy as np
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
        self.index_lock = asyncio.Lock()  # Faiss 인덱스 접근 동기화용
        self.vector_id_to_article_id = {}

    async def _ensure_model_loaded(self):
        """모델이 로드되지 않았다면 비동기로 로드"""
        if self.model is None:
            print("SBERT 모델을 로드합니다...")
            # run_in_threadpool에 callable을 명시적으로 넘김
            def _load():
                return SentenceTransformer('jhgan/ko-sroberta-multitask')
            self.model = await run_in_threadpool(_load)

    async def encode_text(self, text: str) -> np.ndarray:
        """SBERT 임베딩을 비동기 실행 (스레드풀 사용)"""
        await self._ensure_model_loaded()
        # model.encode는 블로킹일 수 있으니 threadpool로 실행
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

        # 1️⃣ 모든 추천 벡터와 ID 가져오기
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

        # 2️⃣ numpy 변환 및 정규화
        vectors_np = np.array(all_vectors, dtype='float32')
        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, vectors_np)

            start_idx = self.index.ntotal
            await run_in_threadpool(self.index.add, vectors_np)

            # 3️⃣ reco_id 매핑
            for i, reco_id in enumerate(reco_ids):
                self.index_to_reco_id[start_idx + i] = reco_id

            # 4️⃣ article_id 매핑
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
            # 여러 벡터 추가될 수 있으므로 range로 처리 (여기선 1개)
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

        # 문자열로 저장된 벡터를 파싱 (TEXT 컬럼 대응)
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

        # 벡터 형태 디버깅 출력
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

            # 🚨 잘못된 차원 방어
            if user_arr.shape[1] != self.index.d:
                print(f"[경고] 잘못된 벡터 차원 발견 ({user_arr.shape[1]} != {self.index.d}) → 필터링")
                # index.d와 같은 차원만 남김
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

        # 4️⃣ 인덱스 상태 확인
        print(f"[DEBUG] index.ntotal={self.index.ntotal}, index.d={getattr(self.index, 'd', None)}")

        if getattr(self.index, "d", None) is None:
            print("[ERROR] 인덱스가 초기화되지 않았습니다. load_and_build_index() 실행 필요.")
            return []

        # 5️⃣ 차원 불일치 확인
        if user_profile.shape[1] != self.index.d:
            raise RuntimeError(
                f"[차원 불일치] user_profile={user_profile.shape[1]} / index.d={self.index.d} "
                f"→ DB 벡터 또는 모델 임베딩 차원 불일치. 인덱스 재생성 필요."
            )

        # 6️⃣ 검색 (Faiss는 동기, threadpool로 실행)
        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, user_profile)
            num_search = top_k + len(user_vectors)
            D, I = await run_in_threadpool(self.index.search, user_profile, num_search)

        similar_reco_ids = []
        for faiss_index_id in I[0]:
            reco_id = self.index_to_reco_id.get(faiss_index_id)
            if reco_id:
                similar_reco_ids.append(reco_id)

        # 7️⃣ 이미 읽은 기사 제외
        read_reco_ids_query = (
            select(Article.article_recommend_id)
            .join(UserRecentArticle, UserRecentArticle.article_id == Article.id)
            .where(UserRecentArticle.user_id == user_id)
        )
        read_reco_ids_result = await db.execute(read_reco_ids_query)
        read_reco_ids = set(read_reco_ids_result.scalars().all())

        filtered_reco_ids = [rid for rid in similar_reco_ids if rid not in read_reco_ids]
        if not filtered_reco_ids:
            print(f"[DEBUG] 추천 가능한 새 기사 없음. (모두 이미 읽음)")
            return []

        # 8️⃣ 추천 기사 ID 반환
        query_similar_articles = (
            select(Article.id)
            .where(Article.article_recommend_id.in_(filtered_reco_ids))
            .limit(top_k)
        )
        result = await db.execute(query_similar_articles)
        recommended_article_ids = result.scalars().all()

        print(f"[DEBUG] 추천된 기사 ID 목록: {recommended_article_ids}\n")

        return recommended_article_ids

    async def find_similar_documents_by_article(
        self, db: AsyncSession, article_id: int, top_k: int = 5
    ) -> list[int]:
        """
        주어진 기사(article_id)와 유사한 기사들을 Faiss 인덱스를 이용해 찾음
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

        # 2️⃣ FAISS 인덱스에서 유사 벡터 검색
        D, I = self.index.search(np.array([query_vector]), top_k + 1)  # +1: 자기 자신 포함
        similar_indices = I[0][1:].tolist()  # 첫 번째(자기 자신) 제외

        # 3️⃣ ID 매핑 (벡터 인덱스 → article_id)
        similar_article_ids = [
            self.vector_id_to_article_id[idx] for idx in similar_indices if idx in self.vector_id_to_article_id
        ]

        return similar_article_ids

# 싱글톤 인스턴스
analysis_service = AnalysisService()