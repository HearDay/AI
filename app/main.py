from fastapi import FastAPI
from app.core.database import engine, Base, SessionLocal
# 👇👇👇 이 부분을 수정합니다! (models -> document)
from app.models import document 
from app.api.endpoints import documents as recommend_router 
from app.services.analysis_service import analysis_service

app = FastAPI(title="LLM & SBERT 기반 텍스트 분석 API")

@app.on_event("startup")
async def on_startup():
    # 1. DB 테이블 생성 (document.py의 모든 테이블)
    async with engine.begin() as conn:
        # 👇👇👇 이 부분도 수정합니다! (Base가 document.Base에 연결됨)
        await conn.run_sync(Base.metadata.create_all) 
    
    # 2. Faiss 인덱스 빌드
    async with SessionLocal() as session:
        await analysis_service.load_and_build_index(session)

# 라우터 포함
app.include_router(recommend_router.router)

@app.get("/")
def read_root():
    return {"message": "AI 추천 API 서버에 오신 것을 환영합니다."}