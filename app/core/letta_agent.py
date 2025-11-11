from typing import Any, Dict, Tuple
from letta.agent import Agent

_AGENTS: Dict[Tuple[str, str], Agent] = {}


def build_agent(user_id: str, session_id: str, system_prompt: str) -> Agent:
    """새로운 LettA Agent 인스턴스 생성"""
    agent = Agent()
    agent.user_id = user_id
    agent.session_id = session_id
    agent.system_prompt = system_prompt
    agent.persona = "curious_mentor"
    return agent


def get_agent(user_id: str, session_id: str, system_prompt: str) -> Agent:
    """유저/세션 조합별 LettA Agent 재사용"""
    key = (user_id, session_id)
    if key not in _AGENTS:
        _AGENTS[key] = build_agent(user_id, session_id, system_prompt)
    return _AGENTS[key]


def safe_chat(agent: Agent, message: str) -> dict:
    """LettA Agent 대화 안전 실행"""
    try:
        reply = agent.chat(message)

        # dict형 응답이면 텍스트만 추출
        if isinstance(reply, dict) and "text" in reply:
            reply = reply["text"]

        return {"answer": str(reply).strip(), "fallback": False}

    except Exception as e:
        return {"answer": f"(임시 응답) LettA 처리 중 오류 발생: {e}", "fallback": True}
