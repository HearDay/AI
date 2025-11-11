from typing import Any, Dict, Tuple
from letta.agent import Agent
from letta.orm import init_db
import asyncio

# -----------------------------------------------------------
# LettA 런타임 관리
# -----------------------------------------------------------
_AGENTS: Dict[Tuple[str, str], Agent] = {}
_DB_INITIALIZED = False


async def _ensure_db_ready():
    """LettA 내부 SQLite DB 초기화 (최초 1회만 실행)"""
    global _DB_INITIALIZED
    if not _DB_INITIALIZED:
        try:
            await init_db()
            _DB_INITIALIZED = True
            print("[LettA] 내부 데이터베이스 초기화 완료.")
        except Exception as e:
            print(f"[LettA] DB 초기화 중 오류: {e}")


def build_agent(user_id: str, session_id: str, system_prompt: str) -> Agent:
    """LettA Agent 인스턴스 생성"""
    # LettA Agent는 직접 __init__ 호출 가능 (user/session 식별자 직접 속성으로 지정)
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
        # LettA ORM 초기화는 비동기이므로 필요 시 실행
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_ensure_db_ready())
            else:
                loop.run_until_complete(_ensure_db_ready())
        except RuntimeError:
            asyncio.run(_ensure_db_ready())

        # 대화 수행
        reply = agent.chat(message)

        # dict형 응답이면 텍스트만 추출
        if isinstance(reply, dict) and "text" in reply:
            reply = reply["text"]

        return {"answer": str(reply).strip(), "fallback": False}

    except Exception as e:
        return {"answer": f"(임시 응답) LettA 처리 중 오류: {e}", "fallback": True}
