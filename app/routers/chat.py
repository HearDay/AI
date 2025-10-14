from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# 프롬프트 템플릿에서 톤/레벨 가이드 가져오기
from app.prompt_templates import CONVERSATIONAL_STYLE, LEVEL_GUIDES

# MemGPT 에이전트 제어
from app.memgpt_agent import get_agent, reset_agent, safe_chat

router = APIRouter(tags=["chat"])

# ---------- 입력 스키마 ----------
class ChatReq(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., description="사용자 메시지")
    level: Optional[str] = Field("beginner", description="beginner|intermediate|advanced")

class ResetReq(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    session_id: str = Field(..., description="세션 ID")
    level: Optional[str] = Field("beginner", description="beginner|intermediate|advanced")

# ---------- 시스템 프롬프트 조립 ----------
def make_system_prompt(level: str) -> str:
    guide = LEVEL_GUIDES.get((level or "beginner").lower(), LEVEL_GUIDES["beginner"])
    return (
        "너는 뉴스 토론 파트너다.\n"
        f"{CONVERSATIONAL_STYLE}\n"
        f"{guide}\n"
        "[작업]\n"
        "- 사용자의 질문에 자연스럽게 응답하고,\n"
        "- 맥락을 기억하며, 필요하면 반문이나 탐구형 질문을 덧붙인다.\n"
    )

# ---------- 엔드포인트 ----------
@router.post("/chat")
def chat(req: ChatReq):
    try:
        system_prompt = make_system_prompt(req.level or "beginner")
        
        agent = get_agent(req.user_id, req.session_id, system_prompt)
        result = safe_chat(agent, req.message)
        return {"answer": result["answer"], "used_memory": True, "fallback": result["fallback"]}
    except RuntimeError as e:
        # MemGPT 미설치/임포트 실패 → 503로 안내
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat error: {e}")

@router.post("/chat/reset")
def chat_reset(req: ResetReq):
    try:
        system_prompt = make_system_prompt(req.level or "beginner")
        reset_agent(req.user_id, req.session_id, system_prompt)
        return {"ok": True}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset error: {e}")
