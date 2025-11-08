from fastapi import APIRouter, HTTPException
from app.core.schemas import SummarizeIn, SummarizeOut, KeywordsIn, KeywordsOut
from app.services.summarize import SummarizeService
from app.services.llm import LLMClient
from app.core.memgpt_client import MemGPTClient

router = APIRouter(prefix="", tags=["summary"])

# Lazy initialization
_llm = None
_mem = None
_service = None

def get_service():
    """Lazy load service on first request"""
    global _llm, _mem, _service
    if _service is None:
        _llm = LLMClient()
        _mem = MemGPTClient()
        _service = SummarizeService(_llm, _mem)
    return _service

@router.post("/summarize", response_model=SummarizeOut)
def post_summarize(payload: SummarizeIn):
    try:
        service = get_service()
        data = service.summarize(payload.user_id, payload.text, payload.level)
        return SummarizeOut(
            summary=data["summary"],
            open_questions=data["open_questions"],
            keywords=data["keywords"]
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable or bad output: {e}")

@router.post("/keywords", response_model=KeywordsOut)
def post_keywords(payload: KeywordsIn):
    service = get_service()
    return KeywordsOut(keywords=service.keywords(payload.text, payload.top_k))
