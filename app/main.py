from fastapi import FastAPI, Form
import asyncio
from app.core.database import engine, Base, SessionLocal
from app.models import document 
from app.api.endpoints import documents as recommend_router 
from app.services.analysis_service import analysis_service
from app.core.prompt_templates import build_open_question_prompt
from app.services.question_generator import generate_question
from app.services import feedback, summary
from app.services.llm import LLMClient  

# ======================================================
#  앱 초기화
# ======================================================
app = FastAPI(title="Hearday AI 토론 & 추천 시스템")

@app.on_event("startup")
async def on_startup():
    # 1. DB 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 2. Faiss 인덱스 빌드 (analysis_service.py)
    async with SessionLocal() as session:
        await analysis_service.load_and_build_index(session)
    
    print("서버 시작 완료. LLM은 첫 요청 시 로드됩니다.")


# ======================================================
# 라우터 등록
# ======================================================
app.include_router(recommend_router.router)
app.include_router(feedback.router)
app.include_router(summary.router)

# ======================================================
# 기본 라우트
# ======================================================
@app.get("/")
def read_root():
    return {"message": "Hearday AI 토론 & 추천 API 서버에 오신 것을 환영합니다."}

@app.get("/health")
def health():
    """서버 상태 확인"""
    return {"ok": True, "status": "running"}

# ======================================================
# AI 토론 관련 엔드포인트
# ======================================================
@app.post("/prompt/preview")
def prompt_preview(level: str = Form("beginner"), summary: str = Form(...)):
    """탐구형 질문 프롬프트 미리보기"""
    return {"level": level, "prompt": build_open_question_prompt(summary, level)}

@app.post("/prompt/question")
def prompt_question(
    mode: str = Form("open"),
    level: str = Form("beginner"),
    context: str = Form(...),
):
    """LLM 기반 뉴스 토론 질문 생성"""
    question = generate_question(context, mode=mode, level=level)
    return {"mode": mode, "level": level, "question": question}
