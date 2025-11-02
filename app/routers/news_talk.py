from fastapi import APIRouter, HTTPException
from app.schemas import PromptQuestionIn
from app.modules.question_generator import generate_question

router = APIRouter(prefix="/prompt", tags=["news_talk"])

@router.post("/question")
def make_news_question(payload: PromptQuestionIn):
    """
    뉴스 요약(content) 또는 대화(message)를 입력받아
    LLM 기반 탐구형 질문 또는 후속질문을 생성
    """
    try:
        # 문맥 조립
        context = payload.content
        if payload.message:
            context = f"{context}\n\n{payload.message}"

        # LLM 호출
        question = generate_question(
            context=context,
            level=payload.level,
            mode=payload.mode
        )

        # reply 

        return {
            "reply": question,
            "mode": payload.mode,
            "level": payload.level,
            "discussionId": payload.discussionId,
            "nickname": payload.nickname,
            "articleTitle": payload.articleTitle
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate question: {e}")
