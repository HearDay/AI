from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from app.memgpt_agent import get_agent, reset_agent, safe_chat

router = APIRouter(tags=["chat"])

class ChatReq(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., description="사용자 메시지")
    level: Optional[str] = Field(None, description="beginner|intermediate|advanced (옵션)")

@router.post("/chat")
def chat(req: ChatReq):
    # level이 넘어오면 레벨별 시스템 프롬프트 적용된 에이전트가 세팅됨
    agent = get_agent(req.user_id, req.session_id, req.level or "beginner")
    # 필요 시 여기서 message 전처리해서 넘겨도 됨 (ex. 레벨 태그 붙이기)
    result = safe_chat(agent, req.message)
    return {"answer": result["answer"], "used_memory": True, "fallback": result["fallback"]}

class ResetReq(BaseModel):
    user_id: str
    session_id: str
    level: Optional[str] = "beginner"

@router.post("/chat/reset")
def chat_reset(req: ResetReq):
    reset_agent(req.user_id, req.session_id, req.level or "beginner")
    return {"ok": True}
