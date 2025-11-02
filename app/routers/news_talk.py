from fastapi import APIRouter, HTTPException
from app.schemas import PromptQuestionIn, PromptQuestionOut
from app.modules.question_generator import generate_question

router = APIRouter(prefix="/prompt", tags=["news_talk"])

@router.post("/question", response_model=PromptQuestionOut)
def make_news_question(payload: PromptQuestionIn):
    """
    뉴스 요약(content) 또는 대화(message)를 입력받아
    LLM 기반 탐구형 질문 또는 후속질문을 생성

    """
    try:
        
        context = payload.content
        if payload.message:
            context = f"{context}\n\n{payload.message}"

        # generate_question 함수 호출 (Kanana 모델 기반)
        question = generate_question(
            context=context,
            level=payload.level,
            mode=payload.mode
        )

        return PromptQuestionOut(
            question=question,
            mode=payload.mode,
            level=payload.level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate question: {e}")
