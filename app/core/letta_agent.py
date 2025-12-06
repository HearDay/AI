from typing import Dict, Tuple
from app.services.llm_memory_bridge import HybridMemoryAgent

# 세션별 HybridMemoryAgent 캐시
_AGENTS: Dict[Tuple[str, str], HybridMemoryAgent] = {}


def get_agent(user_id: str, session_id: str, system_prompt: str) -> HybridMemoryAgent:
    """
    유저/세션 조합별로 메모리 에이전트를 하나씩 유지
    - system_prompt는 현재 구조에선 직접 쓰진 않지만, 필요하면 히스토리에 첫 메시지로 넣거나 확장 가능
    """
    key = (user_id, session_id)
    if key not in _AGENTS:
        _AGENTS[key] = HybridMemoryAgent(name=f"{user_id}_{session_id}")
        print(f"[HybridMemory] session created: {user_id}_{session_id}")
    return _AGENTS[key]


def safe_chat(agent: HybridMemoryAgent, message: str) -> dict:
    """
    HybridMemoryAgent의 chat을 감싸는 안전 래퍼
    - 예외 발생 시 fallback 메시지로 대체
    """
    try:
        reply = agent.chat(message)
        return {"answer": reply, "fallback": False}
    except Exception as e:
        print(f"[HybridMemory Error] {e}")
        return {
            "answer": "죄송해요, 지금은 대화를 제대로 처리하지 못했어요. 다시 한 번 말씀해 주실 수 있을까요?",
            "fallback": True,
        }