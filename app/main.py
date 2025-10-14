from fastapi import FastAPI, Form

# 기존 프롬프트/오픈질문 모듈
from app.prompt_templates import build_open_question_prompt
from app.modules.open_questions import generate_open_questions

# 새로 추가된 라우터 (요약 + 피드백)
from app.routers import feedback as feedback_router
from app.routers import summary as summary_router
from app.routers.chat import router as chat_router


# ---------------------------------------
# 앱 초기화
# ---------------------------------------
app = FastAPI(title="Debate Prompt & Open Questions", version="0.2.0")


# ---------------------------------------
# 기본 라우트
# ---------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


# ---------------------------------------
# 기존 prompt endpoints
# ---------------------------------------
@app.post("/prompt/preview")
def prompt_preview(level: str = Form("beginner"), summary: str = Form(...)):
    return {"level": level, "prompt": build_open_question_prompt(summary, level)}


@app.post("/prompt/open_questions")
def prompt_open_questions(level: str = Form("beginner"), summary: str = Form(...)):
    return {"level": level, "open_questions": generate_open_questions(summary, level)}


# ---------------------------------------
# 새로 추가한 기능
# ---------------------------------------
# 후속질문/피드백
app.include_router(feedback_router.router)

# 요약 + 개방형 질문 + 키워드
app.include_router(summary_router.router)

# 기존 chat 라우터
app.include_router(chat_router)
