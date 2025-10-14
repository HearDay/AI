from fastapi import APIRouter, HTTPException
from app.schemas import FeedbackIn, FeedbackOut
from app.services.followup import FollowupService
from app.core.llm import LLMClient
from app.core.memgpt_client import MemGPTClient

router = APIRouter(prefix="/feedback", tags=["feedback"])

_llm = LLMClient()
_mem = MemGPTClient()
_service = FollowupService(_llm, _mem)

@router.post("", response_model=FeedbackOut)
def create_feedback(payload: FeedbackIn):
    try:
        data = _service.run(
            payload.user_id,
            payload.question,
            payload.user_answer,
            payload.rubric or "",
            payload.level
        )
        return FeedbackOut(**data)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable or bad output: {e}")
