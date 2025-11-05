from typing import Any, Dict, Tuple, Optional

# -----------------------------------------------------------
# MemGPT 다중 경로 임포트 (버전별 구조 차이 대응)
# -----------------------------------------------------------
Agent: Optional[Any] = None
MemoryStore: Optional[Any] = None
_get_provider: Optional[Any] = None
_memgpt_import_error: Optional[Exception] = None

try:
    from memgpt.agent import Agent  # type: ignore
    from memgpt.memory import MemoryStore  # type: ignore
    from memgpt.providers import get_provider as _get_provider  # type: ignore
except ImportError as e1:
    try:
        from memgpt.agent.agent import Agent  # type: ignore
        from memgpt.memory.memory import MemoryStore  # type: ignore
        from memgpt.providers import get_provider as _get_provider  # type: ignore
    except ImportError as e2:
        try:
            from memgpt.core.agent import Agent  # type: ignore
            from memgpt.core.memory import MemoryStore  # type: ignore
            from memgpt.core.providers import get_provider as _get_provider  # type: ignore
        except ImportError as e3:
            _memgpt_import_error = e3


def ensure_memgpt() -> None:
    """MemGPT가 로드되지 않았다면 사용자에게 설치/버전 안내."""
    if Agent is None or MemoryStore is None or _get_provider is None:
        raise RuntimeError(
            "MemGPT 모듈을 찾을 수 없습니다.\n"
            "해결 방법:\n"
            "  1) pip install -U memgpt\n"
            "  2) 또는 git clone https://github.com/cpacker/MemGPT.git && pip install -e ./MemGPT\n\n"
            f"원인: {repr(_memgpt_import_error)}"
        )


# -----------------------------------------------------------
# MemGPT 런타임 초기화 및 관리
# -----------------------------------------------------------
_PROVIDER: Optional[Any] = None
_MEMORY_STORE: Optional[Any] = None
_AGENTS: Dict[Tuple[str, str], Any] = {}


def _ensure_runtime_ready() -> None:
    """Provider/MemoryStore를 최초 접근 시 지연 초기화."""
    ensure_memgpt()
    global _PROVIDER, _MEMORY_STORE
    if _PROVIDER is None:
        _PROVIDER = _get_provider()
    if _MEMORY_STORE is None:
        if hasattr(MemoryStore, "from_default"):
            _MEMORY_STORE = MemoryStore.from_default()
        else:
            _MEMORY_STORE = MemoryStore()


def build_agent(user_id: str, session_id: str, system_prompt: str) -> Any:
    """
    세션별 Agent를 생성.
    - system_prompt는 외부에서 주입(prompt_templates 등)
    """
    _ensure_runtime_ready()
    return Agent(
        provider=_PROVIDER,
        memory_store=_MEMORY_STORE,
        user_id=user_id,
        session_id=session_id,
        system_prompt=system_prompt,
        persona="curious_mentor",
    )


def get_agent(user_id: str, session_id: str, system_prompt: str) -> Any:
    """기존 Agent를 반환하거나 없으면 새로 생성."""
    key = (user_id, session_id)
    if key not in _AGENTS:
        _AGENTS[key] = build_agent(user_id, session_id, system_prompt)
    return _AGENTS[key]


def reset_agent(user_id: str, session_id: str, system_prompt: str) -> None:
    """세션의 Agent를 재생성."""
    key = (user_id, session_id)
    _AGENTS[key] = build_agent(user_id, session_id, system_prompt)


def safe_chat(agent: Any, message: str) -> dict:
    """
    Agent에 안전하게 질의.
    - MemGPT 미설치/로드 실패 시 RuntimeError 발생 (라우터에서 503 처리)
    - 내부 예외 발생 시 fallback 응답 반환 (서버 중단 방지)
    """
    ensure_memgpt()
    try:
        reply = agent.chat(message)
        return {"answer": reply, "fallback": False}
    except Exception as e:
        return {
            "answer": "요청이 길거나 모델이 바쁩니다. 임시 응답을 제공합니다.",
            "fallback": True,
            "error": str(e),
        }
