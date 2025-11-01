from fastapi import APIRouter, Form
from app.services.news_talk import make_question

router = APIRouter()

@router.post("/ask")
async def ask_news_talk(
    summary: str = Form(...),
    opinion: str = Form("")
):
    """
    뉴스 요약 + 사용자 의견 → 탐구형 질문 생성
    """
    question = await make_question(summary, opinion)
    return {"question": question}
