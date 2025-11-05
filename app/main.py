from fastapi import FastAPI, Form
from app.prompt_templates import build_open_question_prompt
from app.modules.question_generator import generate_question

# ---------------------------------------
# 라우터 임포트
# ---------------------------------------
from app.routers import (
    voice,
    feedback as feedback_router,
    summary as summary_router,
    news_talk as news_talk_router,
)
from app.routers.chat import router as chat_router
from app.routers import discussion_api as discussion_router
# ---------------------------------------
# 앱 초기화
# ---------------------------------------
app = FastAPI(title="Debate Prompt & Question Generator", version="0.3.0")

# ---------------------------------------
# 기본 라우트
# ---------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

# ---------------------------------------
# 질문 관련 엔드포인트
# ---------------------------------------
@app.post("/prompt/preview")
def prompt_preview(level: str = Form("beginner"), summary: str = Form(...)):
    """
    프롬프트 미리보기 (탐구형 질문용)
    """
    return {"level": level, "prompt": build_open_question_prompt(summary, level)}

@app.post("/prompt/question")
def prompt_question(
    mode: str = Form("open"),  # open or followup
    level: str = Form("beginner"),
    context: str = Form(...),
):
    """
    LLM 기반 질문 생성
    """
    question = generate_question(context, mode=mode, level=level)
    return {
        "mode": mode,
        "level": level,
        "question": question
    }

app.include_router(feedback_router.router)
app.include_router(summary_router.router)
app.include_router(chat_router)
app.include_router(news_talk_router.router)
app.include_router(voice.router)
app.include_router(discussion_router.router)
