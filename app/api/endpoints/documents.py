from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from pydantic import BaseModel, Field
import datetime

# DB, 모델, 서비스 모듈 임포트
from app.core.database import get_db
from app.models.document import Document as DocumentModel
from app.services.keyword_extractor import keyword_extractor
from app.services.analysis_service import analysis_service # Faiss가 적용된 서비스

# --- Pydantic 스키마 (API 입출력 형식 정의) ---

# [참고]
# app/models/document.py 파일도 백엔드에서 받는 데이터에 맞춰
# article_id, original_url, published_at, title, category 컬럼이
# 추가되어 있어야 합니다!

class DocumentCreate(BaseModel):
    """
    백엔드(크롤러)로부터 받을 데이터의 형식
    """
    article_id: str = Field(..., description="뉴스 원본의 고유 ID")
    original_url: str = Field(..., description="뉴스 원본 URL")
    published_at: datetime.datetime = Field(..., description="발행 시간")
    title: str = Field(..., description="기사 제목")
    text: str = Field(..., description="기사 본문")
    category: str = Field(..., description="뉴스 사이트의 원본 카테고리")

class DocumentResponse(BaseModel):
    """
    API가 성공적으로 문서를 생성한 후 반환할 데이터 형식
    """
    id: int # 우리 DB의 고유 ID
    article_id: str
    title: str
    keywords: Optional[List[str]] = None
    
    class Config:
        from_attributes = True # SQLAlchemy 모델 -> Pydantic 변환

class SimilarDocumentResponse(BaseModel):
    """
    유사 문서 조회 시 반환할 데이터 형식
    """
    doc: DocumentResponse
    score: float

# --- APIRouter 객체 생성 ---
router = APIRouter()

# --- 표준 카테고리 목록 (AI 서버가 내부적으로 관리) ---
# LLM 키워드 추출을 위한 표준 '보기' 목록
STANDARD_CANDIDATES = [
    "인공지능", "IT", "기술", "과학", "경제", "경영", 
    "사회", "정치", "국제", "스포츠", "연예", "문화",
    "농구", "축구", "야구", "반도체", "구글", "애플", "삼성전자"
]


# --- API 엔드포인트 정의 ---

@router.post(
    "/documents", 
    response_model=DocumentResponse, 
    status_code=status.HTTP_201_CREATED,
    summary="새 문서 생성, 분석 및 인덱싱"
)
async def create_document(
    request: DocumentCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    백엔드로부터 새 기사 정보를 받아 중복 여부를 확인한 후,
    LLM 키워드 추출, SBERT 벡터화를 수행하고 DB 및 Faiss 인덱스에 저장합니다.
    """
    
    # 1. 중복 기사 확인 (가장 중요한 방어 로직)
    # article_id (뉴스 고유 ID)를 기준으로 이미 저장되었는지 확인
    query = select(DocumentModel).where(DocumentModel.article_id == request.article_id)
    result = await db.execute(query)
    existing_document = result.scalars().first()
    
    if existing_document:
        # 이미 처리된 기사라면, 200 OK와 함께 기존 정보를 반환
        return existing_document

    # 2. LLM 키워드 추출 (표준 카테고리 사용)
    # 원본 카테고리 + 표준 목록을 합쳐서 후보로 사용할 수도 있습니다.
    # 여기서는 간단하게 표준 목록만 사용합니다.
    keywords = keyword_extractor.extract(request.text, STANDARD_CANDIDATES) 
    
    # 3. SBERT 벡터 생성 (ndarray 반환)
    sbert_vector_np = analysis_service.encode_text(request.text)
    sbert_vector_list = sbert_vector_np.tolist() # DB 저장을 위해 list로 변환

    # 4. DB에 저장할 객체 생성
    new_document = DocumentModel(
        article_id=request.article_id,
        original_url=request.original_url,
        published_at=request.published_at,
        title=request.title,
        text=request.text,
        category=request.category,
        keywords=keywords,
        sbert_vector=sbert_vector_list
    )

    # 5. DB에 저장
    db.add(new_document)
    await db.commit()
    await db.refresh(new_document) # DB ID(new_document.id)를 확정받음
    
    # 6. Faiss 인덱스에 실시간 추가
    await analysis_service.add_document_to_index(
        doc_id=new_document.id, 
        vector_list=sbert_vector_list
    )
    
    # 201 Created 상태 코드와 함께 새로 생성된 정보를 반환
    return new_document

@router.get(
    "/documents/{doc_id}/similar", 
    response_model=List[SimilarDocumentResponse], 
    summary="유사 문서 조회 (Faiss 사용)"
)
async def get_similar_documents(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    특정 문서(doc_id)와 의미적으로 가장 유사한 문서들을
    Faiss 인덱스에서 초고속으로 검색하여 반환합니다.
    """
    
    # Faiss 검색 로직은 모두 analysis_service 내부에 숨겨져 있습니다.
    similar_docs = await analysis_service.find_similar_documents(db, doc_id)
    
    if similar_docs is None:
        raise HTTPException(status_code=404, detail="해당 ID의 문서를 찾을 수 없거나 벡터가 없습니다.")
        
    return similar_docs