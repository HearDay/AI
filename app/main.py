from fastapi import FastAPI
from app.core.database import engine, Base, SessionLocal # SessionLocal 임포트
from app.api.endpoints import documents
from app.services.analysis_service import analysis_service # 서비스 임포트

app = FastAPI(title="LLM & SBERT 기반 텍스트 분석 API")

@app.on_event("startup")
async def on_startup():
    # 1. DB 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 2. Faiss 인덱스 빌드 (매우 중요!)
    # 별도 세션을 생성하여 인덱스 빌드 함수에 주입
    async with SessionLocal() as session:
        await analysis_service.load_and_build_index(session)

app.include_router(documents.router, tags=["Documents"])

# 루트 경로를 간단하게 유지할 수 있습니다.
@app.get("/")
def read_root():
    return {"message": "텍스트 분석 API 서버에 오신 것을 환영합니다."}