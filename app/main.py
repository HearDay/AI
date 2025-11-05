from fastapi import FastAPI, Form
from app.prompt_templates import build_open_question_prompt
from app.modules.question_generator import generate_question

# ---------------------------------------
# 앱 초기화
# ---------------------------------------
app = FastAPI(title="Debate Prompt & Question Generator", version="0.3.0")

# ---------------------------------------
# 라우터 임포트 및 등록
# ---------------------------------------
from app.routers import feedback, summary

app.include_router(feedback.router)
app.include_router(summary.router)

# ---------------------------------------
# 기본 라우트
# ---------------------------------------
@app.get("/health")
def health():
    """서버 상태 확인"""
    return {"ok": True}

# ---------------------------------------
# 프롬프트 미리보기 / 질문 생성 (디버그용)
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
    LLM 기반 탐구형 질문 생성 (단일 요청)
    """
    question = generate_question(context, mode=mode, level=level)
    return {
        "mode": mode,
        "level": level,
        "question": question
    }
