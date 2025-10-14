from typing import Dict, Tuple
from memgpt.agent import Agent
from memgpt.memory import MemoryStore
from memgpt.providers import get_provider


"""
레벨별(초급/중급/심화) 탐구형 질문 템플릿 + '자연스러운 대화 톤' 지침.
- beginner : 사실 이해
- intermediate : 원인/영향
- advanced : 대안/가치판단
"""

LEVEL_GUIDES = {
    "beginner": """[레벨 가이드]
- 사실 이해 중심(누가/언제/어디서/무엇을)
- 문장은 짧고 명료하게
- 유도 질문 금지, 판단 보류
""",
    "intermediate": """[레벨 가이드]
- 원인/영향 중심(왜/누구에게 어떤 영향)
- 이해관계자와 단기/장기 영향 구분
- 반례를 조심스럽게 제안
""",
    "advanced": """[레벨 가이드]
- 대안/가치판단 중심(해결책/원칙/트레이드오프)
- 수용가능한 타협안과 리스크 제시
- 근거 유형(통계/전문가/사례)을 붙임
"""
}

CONVERSATIONAL_STYLE = """[대화 톤 지침]
- 사람처럼 자연스럽게, 과장/명령조 금지.
- 간단한 추임새 허용: "음,", "좋은 포인트예요.", "그렇게 볼 수도 있겠네요."
- 1~2문장 단락, 장황한 수식어 금지, 공손하고 중립.
"""


def build_open_question_prompt(summary: str, level: str = "beginner") -> str:
    lvl = (level or "beginner").lower()
    guide = LEVEL_GUIDES.get(lvl, LEVEL_GUIDES["beginner"])
    return f"""너는 뉴스 토론 파트너다.
{CONVERSATIONAL_STYLE}
{guide}
[작업]
- 아래 뉴스 요약을 읽고, 서로 다른 각도에서 개방형 질문 3개를 만들어라.
- 한 문장씩, 유도/편향 금지. 초보 사용자도 이해 가능한 단어 선택.

[뉴스 요약]
{summary.strip()}

[출력 형식]
1) ...
2) ...
3) ...
"""


def conversational_rewrite(text: str) -> str:
    text = text.replace("해야 한다", "하면 좋겠습니다").replace("하라", "해 볼까요")
    lines = [l.strip(" -•") for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


# -------------------- MemGPT 통합 --------------------

_PROVIDER = get_provider()
_MEMORY_STORE = MemoryStore.from_default()
_AGENTS: Dict[Tuple[str, str], Agent] = {}


def build_agent(user_id: str, session_id: str, level: str = "beginner") -> Agent:
    """
    세션별 Agent 1개 권장.
    level 값에 따라 SYSTEM_PROMPT가 달라진다.
    """
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES["beginner"])
    system_prompt = f"""너는 뉴스 토론 파트너다.
{CONVERSATIONAL_STYLE}
{guide}
[작업]
- 사용자의 질문에 자연스럽게 응답하고,
- 맥락을 기억하며, 필요하면 반문이나 탐구형 질문을 덧붙인다.
"""

    return Agent(
        provider=_PROVIDER,
        memory_store=_MEMORY_STORE,
        user_id=user_id,
        session_id=session_id,
        system_prompt=system_prompt,
        persona="curious_mentor",
    )


def get_agent(user_id: str, session_id: str, level: str = "beginner") -> Agent:
    key = (user_id, session_id)
    if key not in _AGENTS:
        _AGENTS[key] = build_agent(user_id, session_id, level)
    return _AGENTS[key]


def reset_agent(user_id: str, session_id: str, level: str = "beginner") -> None:
    key = (user_id, session_id)
    _AGENTS[key] = build_agent(user_id, session_id, level)


def safe_chat(agent: Agent, message: str) -> dict:
    try:
        reply = agent.chat(message)
        return {"answer": reply, "fallback": False}
    except Exception as e:
        return {
            "answer": "요청이 길거나 모델이 바쁩니다. 임시 응답을 제공합니다.",
            "fallback": True,
            "error": str(e),
        }
