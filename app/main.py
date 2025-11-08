from fastapi import FastAPI, Form
import asyncio
from app.core.database import engine, Base, SessionLocal
from app.models import document
from app.api.endpoints import documents as recommend_router
from app.services.analysis_service import analysis_service
from app.core.prompt_templates import build_open_question_prompt
from app.services.question_generator import generate_question
from app.services import feedback, summary
from app.services.llm import LLMClient  # 모델 미리 로드용

# ======================================================
#  앱 초기화
# ======================================================
app = FastAPI(title="Hearday AI 토론 & 추천 시스템")

# ======================================================
#  Kanana 모델 미리 로드 (시연 속도 개선 핵심)
# ======================================================
@app.on_event("startup")
async def on_startup():
    # 1. DB 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 2. Faiss 인덱스 백그라운드 빌드
    async def _build_faiss():
        async with SessionLocal() as session:
            await analysis_service.load_and_build_index(session)
    asyncio.create_task(_build_faiss())

    # 3. Kanana 모델 미리 로드
    print("Kanana 모델 로드 중... (약 2~3분 소요)")
    global _llm_client
    _llm_client = LLMClient()  # 모델을 전역으로 로드
    print("Kanana 모델 로드 완료. 시연 중 즉시 응답 가능합니다.")


# ======================================================
#  라우터 등록
# ======================================================
app.include_router(recommend_router.router)
app.include_router(feedback.router)
app.include_router(summary.router)

# ======================================================
#  기본 라우트
# ======================================================
@app.get("/")
def read_root():
    return {"message": "Hearday AI 토론 & 추천 API 서버에 오신 것을 환영합니다."}

@app.get("/health")
def health():
    """서버 상태 확인"""
    return {"ok": True, "status": "running"}

# ======================================================
#  AI 토론 관련 엔드포인트
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
