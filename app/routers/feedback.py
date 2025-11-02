from fastapi import APIRouter, HTTPException
from app.schemas import FeedbackIn
from app.modules.question_generator import generate_question
from app.core.memgpt_client import MemGPTClient

router = APIRouter(prefix="/feedback", tags=["feedback"])

_mem = MemGPTClient()  # 세션 기반 대화 기억 유지용

@router.post("")
def create_feedback(payload: FeedbackIn):
    """
    사용자 답변 기반 후속 피드백/질문 생성 (대화 맥락 유지)
    - Kanana 모델 기반 질문 생성
    - MemGPT로 세션별 대화 기록 관리
    """
    try:
        # 1. 현재 대화 기록 불러오기
        past_context = _mem.read_memory(payload.user_id, payload.session_id)

        # 2. 모델에게 전달할 전체 맥락 조립
        context = f"{past_context}\n\n사용자: {payload.user_answer}"

        # 3. 후속 질문 생성 (LLM 호출)
        feedback_text = generate_question(
            context,
            mode="followup",
            level=payload.level or "beginner"
        )

        # 4. 메모리에 새 대화 기록 저장
        _mem.write_event(payload.user_id, "feedback_generated", {
            "question": payload.question,
            "user_answer": payload.user_answer,
            "feedback": feedback_text,
            "level": payload.level
        })

        # 5. 최종 응답 반환
        return {
            "reply": feedback_text,
            "mode": "followup",
            "level": payload.level,
            "question": payload.question,
            "user_answer": payload.user_answer
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM or memory error: {e}")
