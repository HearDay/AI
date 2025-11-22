import asyncio
import faiss
import numpy as np
import json
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi.concurrency import run_in_threadpool
from app.models.document import Article, ArticleRecommend, ArticleRecommendVector, UserRecentArticle

class AnalysisService:
    def __init__(self):
        print("AnalysisService 초기화 중...")
        self.model = None
        self.d = 768
        self.index = faiss.IndexFlatIP(self.d)
        self.index_to_reco_id = {}
        self.index_lock = asyncio.Lock() # API용 비동기 락
        self.vector_id_to_article_id = {}

    def _ensure_model_loaded_sync(self):
        """[동기] 모델 로딩 (백그라운드 작업용)"""
        if self.model is None:
            print("SBERT 모델을 로드합니다... (동기)")
            self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            print("SBERT 모델 로드 완료.")

    async def _ensure_model_loaded(self):
        """[비동기] 모델 로딩 (API용)"""
        if self.model is None:
            await run_in_threadpool(self._ensure_model_loaded_sync)

    # 👇 [수정됨] async def -> def (동기 함수)
    def encode_text(self, text: str) -> np.ndarray:
        # await 없이 직접 동기 함수 호출
        self._ensure_model_loaded_sync()
        # run_in_threadpool 제거하고 직접 호출
        embedding = self.model.encode(text)
        return np.asarray(embedding, dtype='float32')

    # 👇 [수정됨] async def -> def (동기 함수)
    def add_document_to_index(self, reco_id: int, vector_list: list):
        self._ensure_model_loaded_sync()
        vector_np = np.array([vector_list], dtype='float32')
        
        # Faiss 인덱스 추가 (동기 실행)
        faiss.normalize_L2(vector_np)
        start_idx = self.index.ntotal
        self.index.add(vector_np)
        
        for i in range(vector_np.shape[0]):
            self.index_to_reco_id[start_idx + i] = reco_id
            
        print(f"ArticleRecommend ID {reco_id}가 인덱스 {start_idx}에 추가됨")

    # --- 아래는 API용 비동기 함수들 (기존 유지) ---

    async def load_and_build_index(self, db: AsyncSession):
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

        all_vectors = []
        reco_ids = []
        
        for reco_id, vector_data in rows:
            vector_list = None
            if isinstance(vector_data, str):
                try: vector_list = json.loads(vector_data)
                except: continue
            elif isinstance(vector_data, (list, tuple)):
                vector_list = vector_data
            
            if vector_list and len(vector_list) == self.d:
                all_vectors.append(vector_list)
                reco_ids.append(reco_id)

        if not all_vectors:
            return

        vectors_np = np.array(all_vectors, dtype='float32')
        
        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, vectors_np)
            start_idx = self.index.ntotal
            await run_in_threadpool(self.index.add, vectors_np)

            for i, reco_id in enumerate(reco_ids):
                self.index_to_reco_id[start_idx + i] = reco_id

        print(f"총 {self.index.ntotal}개의 벡터가 Faiss 인덱스에 로드되었습니다.")

    async def find_similar_documents_by_user(self, db: AsyncSession, user_id: int, top_k: int = 5) -> list[int]:
        # (사용자 취향 기반 추천 로직 - 기존 코드 유지)
        await self._ensure_model_loaded()
        
        query = (
            select(ArticleRecommendVector.sbert_vector)
            .join(ArticleRecommend)
            .join(Article)
            .join(UserRecentArticle, UserRecentArticle.article_id == Article.id)
            .where(UserRecentArticle.user_id == user_id)
        )
        result = await db.execute(query)
        user_vectors_raw = result.scalars().all()
        
        user_vectors = []
        for v in user_vectors_raw:
            if isinstance(v, (list, tuple)): user_vectors.append(v)
            elif isinstance(v, str): 
                try: user_vectors.append(json.loads(v))
                except: pass
        
        if not user_vectors: return []

        user_arr = np.array(user_vectors, dtype='float32')
        if user_arr.ndim == 1: user_arr = user_arr.reshape(1, -1)

        async with self.index_lock:
            await run_in_threadpool(faiss.normalize_L2, user_arr)
            user_profile = np.mean(user_arr, axis=0).reshape(1, -1)
            await run_in_threadpool(faiss.normalize_L2, user_profile)
            D, I = await run_in_threadpool(self.index.search, user_profile, top_k * 5)

        similar_reco_ids = [self.index_to_reco_id.get(idx) for idx in I[0] if self.index_to_reco_id.get(idx)]
        
        read_reco_ids_query = (
            select(Article.article_recommend_id)
            .join(UserRecentArticle, UserRecentArticle.article_id == Article.id)
            .where(UserRecentArticle.user_id == user_id)
        )
        read_reco_ids_result = await db.execute(read_reco_ids_query)
        read_reco_ids = set(read_reco_ids_result.scalars().all())
        
        filtered_reco_ids = [rid for rid in similar_reco_ids if rid not in read_reco_ids]
        
        if not filtered_reco_ids:
            return []
            
        query_arts = select(Article.id).where(Article.article_recommend_id.in_(filtered_reco_ids)).limit(top_k)
        res = await db.execute(query_arts)
        return res.scalars().all()

    async def find_similar_documents(self, db: AsyncSession, article_id: int, top_k: int = 5) -> list[int]:
         # (기사 기반 추천 로직 - 기존 코드 유지)
        query = select(ArticleRecommendVector.sbert_vector)\
                .join(ArticleRecommend)\
                .join(Article)\
                .where(Article.id == article_id)
        
        result = await db.execute(query)
        vector_row = result.scalar_one_or_none()

        if not vector_row:
            return []

        query_vector = None
        if isinstance(vector_row, str):
            try: query_vector = np.array(json.loads(vector_row), dtype='float32')
            except: pass
        elif isinstance(vector_row, (list, tuple)):
            query_vector = np.array(vector_row, dtype='float32')
            
        if query_vector is None:
            return []

        async with self.index_lock:
            qv = query_vector.reshape(1, -1)
            await run_in_threadpool(faiss.normalize_L2, qv)
            D, I = await run_in_threadpool(self.index.search, qv, top_k * 3)

        similar_reco_ids = []
        for idx in I[0]:
            reco_id = self.index_to_reco_id.get(idx)
            if reco_id:
                similar_reco_ids.append(reco_id)
        
        query_similar_articles = select(Article.id)\
                                 .where(Article.article_recommend_id.in_(similar_reco_ids))\
                                 .where(Article.id != article_id) 
        
        result = await db.execute(query_similar_articles)
        return result.scalars().all()[:top_k]

analysis_service = AnalysisService()