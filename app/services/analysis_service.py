from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.document import Document

class AnalysisService:
    def __init__(self):
        print("SBERT 모델을 로드합니다...")
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        print("SBERT 모델 로드가 완료되었습니다.")

    def encode_text(self, text: str) -> list:
        embedding = self.model.encode(text)
        return embedding.tolist()

    async def find_similar_documents(
        self, db: AsyncSession, doc_id: int, top_k: int = 5
    ):
        query_doc = await db.get(Document, doc_id)
        if not query_doc:
            return None
        
        result = await db.execute(select(Document))
        all_docs = result.scalars().all()

        doc_vectors = [doc.sbert_vector for doc in all_docs]
        query_vector = query_doc.sbert_vector

        similarities = cosine_similarity([query_vector], doc_vectors)[0]

        similar_docs_with_scores = []
        for i, doc in enumerate(all_docs):
            if doc.id != query_doc.id:
                similar_docs_with_scores.append({"doc": doc, "score": similarities[i]})
        
        sorted_docs = sorted(similar_docs_with_scores, key=lambda x: x['score'], reverse=True)
        
        return sorted_docs[:top_k]

analysis_service = AnalysisService()