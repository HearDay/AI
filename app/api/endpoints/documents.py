from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from pydantic import BaseModel, Field
import datetime

# --- 핵심 모듈 임포트 ---
from app.core.database import get_db
from app.models.document import Document as DocumentModel
from app.services.keyword_extractor import keyword_extractor
from app.services.analysis_service import analysis_service # Faiss가 적용된 서비스

# --- APIRouter 객체 생성 ---
router = APIRouter(
    prefix="/documents",  
    tags=["Documents"]    
)

# --- LLM 키워드 추출을 위한 표준 '보기' 목록 ---
STANDARD_CANDIDATES = [
    "경제",
    "방송 / 연예",
    "IT",
    "쇼핑",
    "생활",
    "해외",
    "스포츠",
    "정치"
]

# --- 1. Pydantic 스키마 (API 출력 형식 정의) ---
# DocumentCreate 모델이 삭제되었습니다.

class DocumentResponse(BaseModel):
    """
    [출력] API가 반환할 문서의 기본 형식
    """
    id: int # 우리 DB의 고유 ID
    article_id: str
    title: str
    keywords: Optional[List[str]] = None
    
    class Config:
        from_attributes = True

class SimilarDocumentResponse(BaseModel):
    """
    [출력] GET /.../similar
    유사 문서 조회 시 반환할 데이터 형식
    """
    doc: DocumentResponse
    score: float

# --- 2. API 엔드포인트 정의 ---

# 👇👇👇 이 API가 기존 POST /documents 를 대체합니다! 👇👇👇
@router.post(
    "/process/{doc_id}", 
    response_model=DocumentResponse,
    summary="[백엔드용] 기사 ID를 받아 AI 분석 및 인덱싱 수행"
)
async def process_document_by_id(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    백엔드가 DB에 기사를 저장한 후, 이 API를 호출하여
    해당 ID의 기사에 대한 AI 분석(LLM, SBERT) 및 Faiss 인덱싱을 트리거합니다.
    """
    
    # 1. DB에서 ID로 기사 데이터 조회
    doc = await db.get(DocumentModel, doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="해당 ID의 문서를 찾을 수 없습니다.")
    
    # 2. 이미 처리되었는지 확인
    if doc.status == 'COMPLETED':
        return doc # 이미 완료된 작업이면 그냥 반환

    # 3. LLM 키워드 추출
    keywords = keyword_extractor.extract(doc.text, STANDARD_CANDIDATES) 
    
    # 4. SBERT 벡터 생성
    sbert_vector_np = analysis_service.encode_text(doc.text)
    sbert_vector_list = sbert_vector_np.tolist() 

    # 5. DB 객체 업데이트 (UPDATE)
    doc.keywords = keywords
    doc.sbert_vector = sbert_vector_list
    doc.status = 'COMPLETED' # 상태를 '완료'로 변경

    # 6. DB에 변경 사항 커밋
    await db.commit()
    await db.refresh(doc)
    
    # 7. Faiss 인덱스에 실시간 추가
    await analysis_service.add_document_to_index(
        doc_id=doc.id, 
        vector_list=sbert_vector_list
    )
    
    return doc


@router.get(
    "/{doc_id}/similar", 
    response_model=List[SimilarDocumentResponse], 
    summary="[SBERT 추천] 특정 기사와 유사한 기사 추천 (Faiss)"
)
async def get_similar_documents(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    [SBERT 기반 추천]
    (이 API는 변경 없음)
    """
    
    similar_docs = await analysis_service.find_similar_documents(db, doc_id)
    
    if similar_docs is None:
        raise HTTPException(status_code=404, detail="해당 ID의 문서를 찾을 수 없거나 벡터가 없습니다.")
        
    return similar_docs


@router.get(
    "/category/{category_name}", 
    response_model=List[DocumentResponse], 
    summary="[LLM 추천] 특정 카테고리 기사 목록 (콜드 스타트용)"
)
async def get_documents_by_category(
    category_name: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    [LLM 기반 추천]
    (이 API는 변경 없음)
    """
    
    query = (
        select(DocumentModel)
        .where(DocumentModel.keywords.contains([category_name]))
        .where(DocumentModel.status == 'COMPLETED') # ★분석 완료된 것만 검색★
        .order_by(DocumentModel.published_at.desc())
        .limit(limit)
    )
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    if not documents:
        raise HTTPException(
            status_code=404, 
            detail=f"'{category_name}' 카테고리의 기사를 찾을 수 없습니다."
        )
        
    return documents