from fastapi import APIRouter, Form
from app.modules.question_generator import generate_question

router = APIRouter(prefix="/news_talk", tags=["news_talk"])

@router.post("/question")
def make_news_question(level: str = Form("beginner"), context: str = Form(...)):
    return {"question": generate_question(context, mode="open", level=level)}
