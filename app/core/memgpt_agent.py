from typing import Any, Dict, Tuple, Optional
import sys, os

# -----------------------------------------------------------
# MemGPT 경로 설정 
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMGPT_ROOT = os.path.join(BASE_DIR, "MemGPT")         # MemGPT 루트 폴더
MEMGPT_SRC = os.path.join(MEMGPT_ROOT, "memgpt")        # 실제 코드 폴더

for path in [MEMGPT_ROOT, MEMGPT_SRC]:
    if path not in sys.path:
        sys.path.append(path)

# -----------------------------------------------------------
# MemGPT 다중 경로 임포트 (버전별 구조 차이 대응)
# -----------------------------------------------------------
Agent: Optional[Any] = None
MemoryStore: Optional[Any] = None
_get_provider: Optional[Any] = None
_memgpt_import_error: Optional[Exception] = None

try:
    # 최신 버전 (0.3+)
    from memgpt.agent_store.agent import Agent  # type: ignore
    from memgpt.agent_store.memory import MemoryStore  # type: ignore
    from memgpt.llm_api.providers import get_provider as _get_provider  # type: ignore
except ImportError as e1:
    try:
        # 이전 버전 (0.2.x)
        from memgpt.agent import Agent  # type: ignore
        from memgpt.memory import MemoryStore  # type: ignore
        from memgpt.providers import get_provider as _get_provider  # type: ignore
    except ImportError as e2:
        _memgpt_import_error = e2


# -----------------------------------------------------------
# MemGPT 로드 확인
# -----------------------------------------------------------
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
    """세션별 Agent 생성"""
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
    key = (user_id, session_id)
    if key not in _AGENTS:
        _AGENTS[key] = build_agent(user_id, session_id, system_prompt)
    return _AGENTS[key]


def reset_agent(user_id: str, session_id: str, system_prompt: str) -> None:
    key = (user_id, session_id)
    _AGENTS[key] = build_agent(user_id, session_id, system_prompt)


def safe_chat(agent: Any, message: str) -> dict:
    """Agent에 안전하게 질의"""
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
