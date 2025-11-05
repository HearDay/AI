from fastapi import APIRouter, HTTPException
from app.schemas import SummarizeIn, SummarizeOut, KeywordsIn, KeywordsOut
from app.services.summarize import SummarizeService
from app.core.llm import LLMClient
from app.core.memgpt_client import MemGPTClient

router = APIRouter(prefix="", tags=["summary"])

_llm = LLMClient()
_mem = MemGPTClient()
_service = SummarizeService(_llm, _mem)

@router.post("/summarize", response_model=SummarizeOut)
def post_summarize(payload: SummarizeIn):
    try:
        data = _service.summarize(payload.user_id, payload.text, payload.level)
        return SummarizeOut(
            summary=data["summary"],
            open_questions=data["open_questions"],
            keywords=data["keywords"]
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable or bad output: {e}")

@router.post("/keywords", response_model=KeywordsOut)
def post_keywords(payload: KeywordsIn):
    return KeywordsOut(keywords=_service.keywords(payload.text, payload.top_k))
