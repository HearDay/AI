from fastapi import FastAPI, Form
from app.prompt_templates import build_open_question_prompt
from app.modules.question_generator import generate_question
from app.routers import voice

from app.routers import feedback as feedback_router
from app.routers import summary as summary_router
from app.routers.chat import router as chat_router
from app.routers import news_talk as news_talk_router


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


# ---------------------------------------
# 추가 기능 라우터 등록
# ---------------------------------------
# 후속질문/피드백
app.include_router(feedback_router.router)

# 요약 + 개방형 질문 + 키워드
app.include_router(summary_router.router)

# 기존 chat 라우터
app.include_router(chat_router)

# 뉴스 기반 대화형 질문
app.include_router(news_talk_router.router)

# TTS, STT 지원
app.include_router(voice.router)