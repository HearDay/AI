import sys, os
from fastapi import FastAPI, Form
import asyncio
from app.core.database import engine, Base, SessionLocal
from app.models import document
from app.api.endpoints import documents as recommend_router
from app.services.analysis_service import analysis_service
from app.core.prompt_templates import build_open_question_prompt
from app.services.question_generator import generate_question
from app.services import feedback
from app.services.llm import LLMClient  

# ======================================================
#  앱 초기화
# ======================================================
app = FastAPI(title="Hearday AI 토론 & 추천 시스템")

# 전역 모델 클라이언트 선언
_llm_client = None


@app.on_event("startup")
async def on_startup():
    # 1. DB 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 2. Faiss 인덱스 백그라운드 빌드
    async def _build_faiss_background():
        """비동기 백그라운드에서 Faiss 인덱스 빌드"""
        try:
            print("FAISS 백그라운드 인덱스 빌드 시작...")
            async with SessionLocal() as session:
                await analysis_service.load_and_build_index(session)
            print("FAISS 인덱스 빌드 완료.")
        except Exception as e:
            print(f"FAISS 인덱스 빌드 중 오류 발생: {e}")

    asyncio.create_task(_build_faiss_background())

    # 3. H2O-Danube-1.8B Chat 모델 로드
    async def _load_h2o_model():
        global _llm_client
        print("H2O-Danube-1.8B Chat 모델 로드 중... (약 1~2분 소요)")
        _llm_client = await asyncio.to_thread(LLMClient)
        print("모델 로드 완료. 시연 중 즉시 응답 가능합니다.")

    asyncio.create_task(_load_h2o_model())


# ======================================================
# 라우터 등록
# ======================================================
app.include_router(recommend_router.router)
app.include_router(feedback.router)


# ======================================================
# AI 토론 관련 엔드포인트
# ======================================================
@app.post("/prompt/question")
def prompt_question(
    mode: str = Form("open_question"),
    level: str = Form("beginner"),
    context: str = Form(...),
):
    """LLM 기반 뉴스 토론 질문 생성"""
    question = generate_question(context, mode=mode, level=level)
    return {"mode": mode, "level": level, "question": question}
