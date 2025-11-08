from fastapi import FastAPI
import asyncio
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
    async def _build_faiss():
        async with SessionLocal() as session:
            await analysis_service.load_and_build_index(session)
    asyncio.create_task(_build_faiss())

# 라우터 포함
app.include_router(recommend_router.router)

@app.get("/")
def read_root():
    return {"message": "AI 추천 API 서버에 오신 것을 환영합니다."}

#------------------------------------------------------------------

from fastapi import Form
from app.core.prompt_templates import build_open_question_prompt
from app.services.question_generator import generate_question
from app.services import feedback, summary

# 라우터 등록
app.include_router(feedback.router)
app.include_router(summary.router)

# 헬스체크
@app.get("/health")
def health():
    """서버 상태 확인"""
    return {"ok": True}

# 프롬프트 프리뷰 (디버그용)
@app.post("/prompt/preview")
def prompt_preview(level: str = Form("beginner"), summary: str = Form(...)):
    """프롬프트 미리보기 (탐구형 질문용)"""
    return {"level": level, "prompt": build_open_question_prompt(summary, level)}

# LLM 기반 질문 생성
@app.post("/prompt/question")
def prompt_question(
    mode: str = Form("open"),
    level: str = Form("beginner"),
    context: str = Form(...),
):
    """LLM 기반 질문 생성"""
    question = generate_question(context, mode=mode, level=level)
    return {"mode": mode, "level": level, "question": question}
