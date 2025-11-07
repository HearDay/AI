from fastapi import FastAPI
from app.core.database import engine, Base, SessionLocal
from app.api.endpoints import documents
from app.services.analysis_service import analysis_service # 서비스 임포트

app = FastAPI(title="LLM & SBERT 기반 텍스트 분석 API")

@app.on_event("startup")
async def on_startup():
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
    
    # Faiss 인덱스 빌드는 DB에 있는 데이터를 읽어오는 것이므로 유지합니다.
    async with SessionLocal() as session:
        await analysis_service.load_and_build_index(session)

# 라우터를 포함시킵니다.
app.include_router(documents.router, tags=["Documents"])

@app.get("/")
def read_root():
    return {"message": "AI 텍스트 분석 API 서버"}