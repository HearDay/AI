from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

# DB 및 모델 관련 모듈
from .core.database import engine, Base, get_db
from .models.document import Document as DocumentModel

# services 폴더의 두 서비스를 모두 가져옵니다.
from .services.keyword_extractor import keyword_extractor
from .services.analysis_service import analysis_service

# --- Pydantic 스키마 정의 ---
class DocumentCreate(BaseModel):
    text: str = Field(..., min_length=1, description="분석 및 저장할 텍스트")

class DocumentResponse(BaseModel):
    id: int
    text: str
    keywords: Optional[List[str]] = None # 응답에 키워드도 포함
    
    class Config:
        from_attributes = True

class SimilarDocumentResponse(BaseModel):
    doc: DocumentResponse
    score: float

# --- FastAPI 앱 설정 ---
app = FastAPI(title="LLM & SBERT 기반 텍스트 분석 API")

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- API 엔드포인트 ---
@app.post("/documents", response_model=DocumentResponse, summary="새 문서 생성 (키워드 추출 + 벡터화)")
async def create_document(
    request: DocumentCreate,
    db: AsyncSession = Depends(get_db)
):
    """텍스트를 받아 키워드 추출과 SBERT 벡터화를 모두 수행한 뒤 DB에 저장합니다."""
    # 1. LLM으로 키워드 추출
    candidate_keywords = ["인공지능", "머신러닝", "데이터", "기술", "구글", "모델", "자동차"]
    keywords = keyword_extractor.extract(request.text, candidate_keywords)

    # 2. SBERT 벡터 생성
    sbert_vector = analysis_service.encode_text(request.text)
    
    # 3. DB에 저장할 객체 생성 (키워드와 벡터 모두 포함)
    new_document = DocumentModel(
        text=request.text,
        keywords=keywords,
        sbert_vector=sbert_vector
    )

    db.add(new_document)
    await db.commit()
    await db.refresh(new_document)
    
    return new_document

@app.get("/documents/{doc_id}/similar", response_model=List[SimilarDocumentResponse], summary="유사 문서 조회")
async def get_similar_documents(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """특정 문서와 의미적으로 가장 유사한 문서들을 반환합니다."""
    similar_docs = await analysis_service.find_similar_documents(db, doc_id)
    if similar_docs is None:
        raise HTTPException(status_code=404, detail="해당 ID의 문서를 찾을 수 없습니다.")
    return similar_docs