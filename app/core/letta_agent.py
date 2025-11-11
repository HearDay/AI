from typing import Any, Dict, Tuple
from letta.agent import Agent
from letta.core.interface.default import DefaultInterface
from letta.schemas.agent import AgentState
from letta.orm.user import User
from letta.schemas.llm_config import LLMConfig

_AGENTS: Dict[Tuple[str, str], Agent] = {}


def build_agent(user_id: str, session_id: str, system_prompt: str) -> Agent:
    """LettA Agent 생성 (필수 인자 포함, 안전버전)"""
    # LettA 필수 인자 준비
    interface = DefaultInterface()
    llm_config = LLMConfig(model="gpt-4o-mini", context_window=8192)
    agent_state = AgentState(system_prompt=system_prompt, llm_config=llm_config)
    user = User(id=user_id)

    # Agent 생성
    agent = Agent(interface, agent_state, user)
    agent.persona = "curious_mentor"
    return agent


def get_agent(user_id: str, session_id: str, system_prompt: str) -> Agent:
    """세션별 LettA Agent 재사용"""
    key = (user_id, session_id)
    if key not in _AGENTS:
        _AGENTS[key] = build_agent(user_id, session_id, system_prompt)
    return _AGENTS[key]


def safe_chat(agent: Agent, message: str) -> dict:
    """LettA Agent 대화 안전 실행"""
    try:
        reply = agent.chat(message)

        if isinstance(reply, dict) and "text" in reply:
            reply = reply["text"]

        return {"answer": str(reply).strip(), "fallback": False}

    except Exception as e:
        return {"answer": f"(임시 응답) LettA 처리 중 오류 발생: {e}", "fallback": True}
