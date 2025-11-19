import sys, os
from fastapi import FastAPI, Form, Body
import asyncio
from app.core.database import engine, Base, SessionLocal
from app.models import document 
from app.api.endpoints.documents import recommend_router 
from app.api.endpoints.documents import router as internal_router
from app.services.analysis_service import analysis_service
from app.core.prompt_templates import build_open_question_prompt
from app.services.question_generator import generate_question
from app.services import feedback


#  ======================================================
#  앱 초기화
# ======================================================
app = FastAPI(title="Hearday AI 토론 & 추천 시스템")

# 전역 모델 클라이언트 선언 (유지)
_llm_client = None


@app.on_event("startup")
async def on_startup():
    # 1. DB 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 2. Faiss 인덱스 빌드 (analysis_service.py)
    async with SessionLocal() as session:
        await analysis_service.load_and_build_index(session)



# ======================================================
# 라우터 등록
# ======================================================
app.include_router(internal_router)
app.include_router(recommend_router)
app.include_router(feedback.router)



# ======================================================
# 음성 토론 관련 엔드포인트
# ======================================================
from app.api.endpoints.voice_discussion import router as voice_discussion_b64_router
app.include_router(voice_discussion_b64_router)
