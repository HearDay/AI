import faiss                   # 1. Faiss 임포트
import numpy as np             # 2. Numpy 임포트
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.document import Document

class AnalysisService:
    def __init__(self):
        print("SBERT 모델을 로드합니다...")
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 4. Faiss 인덱스 초기화
        self.d = 768 # SBERT 모델의 벡터 차원 (ko-sroberta-multitask는 768)
        
        # IndexFlatIP: 'IP' (Inner Product)는 코사인 유사도와 유사하게 작동합니다.
        # L2 정규화된 벡터에 IP를 사용하면 코사인 유사도와 같습니다.
        self.index = faiss.IndexFlatIP(self.d) 
        
        # 5. Faiss 인덱스(0,1,2...)와 DB의 document.id(1, 5, 10...)를 맵핑할 딕셔너리
        # Faiss 인덱스 ID -> DB ID
        self.index_to_db_id = {} 

    def encode_text(self, text: str) -> np.ndarray:
        # 6. 벡터를 list가 아닌 numpy.ndarray로 반환 (float32로)
        embedding = self.model.encode(text)
        return embedding.astype('float32')

    async def load_and_build_index(self, db: AsyncSession):
        """
        [★중요★] 서버 시작 시, DB에 있는 모든 벡터를 Faiss 인덱스로 로드합니다.
        """
        print("DB로부터 Faiss 인덱스를 빌드합니다...")
        result = await db.execute(select(Document.id, Document.sbert_vector))
        all_vectors = []
        db_ids = []

        for doc_id, vector_list in result.all():
            if vector_list:
                all_vectors.append(vector_list)
                db_ids.append(doc_id)
        
        if all_vectors:
            vectors_np = np.array(all_vectors).astype('float32')
            # 7. Faiss IndexFlatIP는 L2 정규화된 벡터가 필요합니다.
            faiss.normalize_L2(vectors_np)
            
            # 8. Faiss 인덱스에 벡터 '추가'
            self.index.add(vectors_np)
            
            # 9. 맵핑 정보 저장
            for i, db_id in enumerate(db_ids):
                self.index_to_db_id[i] = db_id # Faiss 인덱스 i번째는 DB ID db_id
            
            print(f"총 {self.index.ntotal}개의 벡터가 Faiss 인덱스에 로드되었습니다.")

    async def add_document_to_index(self, doc_id: int, vector_list: list):
        """
        [★중요★] 새 문서가 생성될 때, 해당 벡터를 Faiss 인덱스에 실시간으로 추가합니다.
        """
        vector_np = np.array([vector_list]).astype('float32')
        faiss.normalize_L2(vector_np) # 정규화
        
        new_index_id = self.index.ntotal # 추가될 Faiss 인덱스 ID (현재 총 개수)
        self.index.add(vector_np) # 인덱스에 추가
        self.index_to_db_id[new_index_id] = doc_id # 맵핑 정보 추가
        print(f"문서 ID {doc_id}가 Faiss 인덱스 {new_index_id}에 추가되었습니다.")

    async def find_similar_documents(
        self, db: AsyncSession, doc_id: int, top_k: int = 5
    ):
        """
        [★핵심★] Faiss 인덱스를 '검색'하여 유사 문서를 찾습니다.
        """
        # 1. 기준 문서 벡터 조회
        query_doc = await db.get(Document, doc_id)
        if not query_doc or not query_doc.sbert_vector:
            return None
        
        query_vector_np = np.array([query_doc.sbert_vector]).astype('float32')
        faiss.normalize_L2(query_vector_np) # 쿼리 벡터도 정규화

        # 2. Faiss '검색' (k+1 하는 이유: 자기 자신도 포함되기 때문)
        # D = 거리(유사도 점수), I = 인덱스 ID
        D, I = self.index.search(query_vector_np, top_k + 1)

        similar_docs_info = []
        
        # 3. Faiss 인덱스 ID(I[0])를 DB ID로 변환
        for i, faiss_index_id in enumerate(I[0]):
            db_id = self.index_to_db_id.get(faiss_index_id)
            
            # 4. 자기 자신(doc_id)은 결과에서 제외
            if db_id is None or db_id == doc_id:
                continue

            # 5. DB에서 실제 문서 정보 조회 (비효율적일 수 있으나 가장 간단한 방법)
            doc = await db.get(Document, db_id)
            if doc:
                similar_docs_info.append({"doc": doc, "score": float(D[0][i])})
        
        return similar_docs_info[:top_k]


# analysis_service 인스턴스를 싱글턴으로 생성 (중요: Faiss 인덱스를 메모리에 유지)
analysis_service = AnalysisService()