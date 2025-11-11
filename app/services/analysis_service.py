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
    async def get_document_embedding(self, db: AsyncSession, article_id: int):
        """
        특정 기사(article_id)의 SBERT 벡터를 DB에서 가져오거나,
        DB에 없으면 기사 제목+본문으로 새 임베딩을 생성하여 반환.
        반환형: numpy.ndarray (dtype float32), shape (d,)
        """
        # 1) DB에 저장된 벡터 조회
        query = (
            select(ArticleRecommendVector.sbert_vector)
            .join(ArticleRecommend, ArticleRecommendVector.article_recommend_id == ArticleRecommend.id)
            .join(Article, Article.article_recommend_id == ArticleRecommend.id)
            .where(Article.id == article_id)
        )
        result = await db.execute(query)
        vector_row = result.scalar_one_or_none()

        import json, pickle

        if vector_row is not None:
            # 문자열로 저장된 JSON 벡터 처리
            if isinstance(vector_row, str):
                try:
                    vec = json.loads(vector_row)
                    return np.array(vec, dtype='float32')
                except Exception:
                    pass
            # pickle/bytes 처리
            if isinstance(vector_row, (bytes, bytearray)):
                try:
                    vec = pickle.loads(vector_row)
                    return np.array(vec, dtype='float32')
                except Exception:
                    pass
            # 이미 list 등으로 온 경우
            try:
                return np.array(vector_row, dtype='float32')
            except Exception:
                pass

        # 2) DB에 벡터가 없으면, 기사 텍스트(title + description)로 임베딩 생성
        article_query = select(Article.title, Article.description).where(Article.id == article_id)
        article_row = (await db.execute(article_query)).one_or_none()
        if not article_row:
            raise ValueError(f"Article ID {article_id} not found.")

        title, description = article_row
        text = f"{title or ''} {description or ''}".strip()
        if not text:
            raise ValueError(f"Article ID {article_id} has no text to encode.")

        # model 로드/임베딩 생성
        embedding = await self.encode_text(text)  # returns np.ndarray dtype float32
        # embedding은 (d,) 혹은 (d,)형태로 기대
        return np.asarray(embedding, dtype='float32').reshape(-1)
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
                    continue
            elif isinstance(v, (list, tuple)):
                parsed_vectors.append(v)
        user_vectors = parsed_vectors

        if not user_vectors:
            return []

        # numpy 변환
        user_arr = np.array(user_vectors, dtype='float32')
        if user_arr.ndim == 1:
            user_arr = user_arr.reshape(1, -1)

        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, user_arr)
            user_profile = np.mean(user_arr, axis=0).reshape(1, -1)
            await run_in_threadpool(faiss.normalize_L2, user_profile)
            num_search = top_k * 3
            D, I = await run_in_threadpool(self.index.search, user_profile, num_search)

        # 5️⃣ 점수와 reco_id 매핑
        scored_reco_ids = []
        for score, idx in zip(D[0], I[0]):
            reco_id = self.index_to_reco_id.get(idx)
            if reco_id:
                scored_reco_ids.append((float(score.squeeze()), reco_id))

        # 내림차순 정렬
        scored_reco_ids.sort(key=lambda x: x[0], reverse=True)

        # 6️⃣ 이미 읽은 기사 제외
        read_reco_ids_query = (
            select(Article.article_recommend_id)
            .join(UserRecentArticle, UserRecentArticle.article_id == Article.id)
            .where(UserRecentArticle.user_id == user_id)
        )
        read_reco_ids_result = await db.execute(read_reco_ids_query)
        read_reco_ids = set(read_reco_ids_result.scalars().all())
        filtered_scored = [(s, rid) for s, rid in scored_reco_ids if rid not in read_reco_ids]

        # 7️⃣ 3일 이내 기사만 필터링
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

        reco_to_article = {reco_id: article_id for article_id, reco_id, _ in recent_articles}
        recommended_article_ids = [
            reco_to_article[rid]
            for _, rid in filtered_scored
            if rid in reco_to_article
        ][:top_k]

        print(f"[DEBUG] 추천된 기사 (점수 포함): {[(aid, round(s,4)) for s, aid in zip([s for s,_ in filtered_scored], recommended_article_ids)]}")
        return recommended_article_ids


    async def find_similar_documents_by_article(
        self, db: AsyncSession, article_id: int, top_k: int = 5
    ) -> list[int]:
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

        async with self.index_lock:
            qv = query_vector.reshape(1, -1)
            await run_in_threadpool(faiss.normalize_L2, qv)
            D, I = await run_in_threadpool(self.index.search, qv, top_k * 2)

        scored_articles = []
        for score, idx in zip(D[0], I[0]):
            if idx in self.vector_id_to_article_id:
                aid = self.vector_id_to_article_id[idx]
                if aid != article_id:
                    scored_articles.append((float(score.squeeze()), aid))

        # 3일 이내 기사 필터링
        three_days_ago = datetime.now() - timedelta(days=3)
        ids = [aid for _, aid in scored_articles]
        query_recent = (
            select(Article.id, Article.created_at)
            .where(
                Article.id.in_(ids),
                Article.created_at >= three_days_ago
            )
        )
        result = await db.execute(query_recent)
        recent_ids = {aid for aid, _ in result.all()}

        similar_article_ids = [
            aid for s, aid in sorted(scored_articles, key=lambda x: x[0], reverse=True)
            if aid in recent_ids
        ][:top_k]

        print(f"[DEBUG] 유사 기사 결과 (점수 내림차순): {similar_article_ids}")
        return similar_article_ids

# 싱글톤 인스턴스
analysis_service = AnalysisService()