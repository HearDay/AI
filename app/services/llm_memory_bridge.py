from letta.agent import Agent
from app.services.llm_api import run_llm

class HybridMemoryAgent:
    """letta 메모리 + 외부 LLM (Upstage API) 결합형"""

    def __init__(self, name: str = "discussion_agent"):
        self.agent = Agent(name=name)
        print(f"[MemGPT] letta memory initialized: {name}")

    def chat(self, user_input: str, role="user") -> str:
        """사용자 발화 → letta 저장 → API LLM 호출 → 응답 저장"""
        # LettA memory에 추가
        self.agent.memory.add_message(role, user_input)

        context = self.agent.memory.get_recent_messages(limit=5)
        formatted_context = []
        for m in context:
            if isinstance(m, dict):
                formatted_context.append(f"{m.get('role')}: {m.get('content')}")
            else:
                formatted_context.append(f"{getattr(m, 'role', 'user')}: {getattr(m, 'content', str(m))}")

        conversation = "\n".join(formatted_context)

        # LLM 호출 (Upstage Solar API)
        reply = run_llm([
            {"role": "system", "content": "너는 뉴스 기반 토론 AI 파트너다. 사용자의 의견을 존중하며 대화를 이어가라."},
            {"role": "user", "content": conversation}
        ])

        # letta memory에 응답 저장
        self.agent.memory.add_message("assistant", reply)
        return reply
