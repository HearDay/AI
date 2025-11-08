import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import asyncio
from fastapi.concurrency import run_in_threadpool  # 1. run_in_threadpool 임포트
from app.models.document import Article, ArticleRecommend, ArticleRecommendKeyword, ArticleRecommendVector

class AnalysisService:
    def __init__(self):
        print("SBERT 모델을 로드합니다...")
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.d = 768 
        self.index = faiss.IndexFlatIP(self.d) 
        self.index_to_reco_id = {} 

    async def encode_text(self, text: str) -> np.ndarray:
        """
        [수정됨] CPU를 막는 model.encode()를 별도 스레드풀에서 실행합니다.
        """
        embedding = await run_in_threadpool(self.model.encode, text)
        return embedding.astype('float32')

    async def load_and_build_index(self, db: AsyncSession):
        """
        (이 함수는 서버 시작 시 1회만 실행되므로, 
         asyncio.to_thread가 필수는 아니지만, 그대로 둡니다.)
        """
        print("DB로부터 Faiss 인덱스를 빌드합니다...")
        
        query = select(
            ArticleRecommendVector.article_recommend_id, 
            ArticleRecommendVector.sbert_vector
        )
                
        result = await db.execute(query)
        all_vectors = []
        reco_ids = []

        for reco_id, vector_list in result.all():
            if vector_list:
                all_vectors.append(vector_list)
                reco_ids.append(reco_id)
        
        if all_vectors:
            vectors_np = np.array(all_vectors).astype('float32')
            await run_in_threadpool(faiss.normalize_L2, vectors_np)
            await run_in_threadpool(self.index.add, vectors_np)
            
            for i, reco_id in enumerate(reco_ids):
                self.index_to_reco_id[i] = reco_id 
            
            print(f"총 {self.index.ntotal}개의 벡터가 Faiss 인덱스에 로드되었습니다.")

    async def add_document_to_index(self, reco_id: int, vector_list: list):
        vector_np = np.array([vector_list]).astype('float32')
        await run_in_threadpool(faiss.normalize_L2, vector_np)
        
        new_index_id = self.index.ntotal
        await run_in_threadpool(self.index.add, vector_np)
        
        self.index_to_reco_id[new_index_id] = reco_id 
        print(f"ArticleRecommend ID {reco_id}가 Faiss 인덱스 {new_index_id}에 추가되었습니다.")

    async def find_similar_documents(
        self, db: AsyncSession, article_id: int, top_k: int = 5
    ) -> list[int]:
        
        query = select(ArticleRecommendVector.sbert_vector)\
                .join(ArticleRecommend)\
                .join(Article)\
                .where(Article.id == article_id)
        
        result = await db.execute(query)
        vector_list = result.scalars().first()

        if not vector_list:
            return []
        
        query_vector_np = np.array([vector_list]).astype('float32')
        await run_in_threadpool(faiss.normalize_L2, query_vector_np)

        D, I = await run_in_threadpool(
            self.index.search, query_vector_np, top_k + 1
        )

        similar_reco_ids = []
        for i, faiss_index_id in enumerate(I[0]):
            reco_id = self.index_to_reco_id.get(faiss_index_id)
            if reco_id:
                similar_reco_ids.append(reco_id)
        
        if not similar_reco_ids:
            return []
            
        query_similar_articles = select(Article.id)\
                                 .where(Article.article_recommend_id.in_(similar_reco_ids))\
                                 .where(Article.id != article_id) 
        
        result = await db.execute(query_similar_articles)
        similar_article_ids = result.scalars().all()
        
        return similar_article_ids[:top_k]

analysis_service = AnalysisService()