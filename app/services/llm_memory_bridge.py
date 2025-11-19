from app.services.llm_api import run_llm

class HybridMemoryAgent:
    """
    간단한 대화 메모리 + Upstage LLM 결합형 에이전트
    """

    def __init__(self, name: str = "discussion_agent"):
        self.name = name
        self.history: list[dict] = []
        print(f"[HybridMemory] initialized: {name}")

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_recent_messages(self, limit: int = 5):
        return self.history[-limit:]

    def chat(self, user_input: str, role: str = "user") -> str:
        self.add_message(role, user_input)

        context = self.get_recent_messages(limit=5)
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in context)

        reply = run_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "너는 사용자의 의견을 바탕으로 짧고 자연스러운 의견을 말하는 토론 파트너다. "
                        "한 문장으로 간결한 의견 또는 관찰만 표현하라. "
                        "질문은 생성하지 마라."
                    ),
                },
                {"role": "user", "content": conversation},
            ],
            max_tokens=80,
            temperature=0.7,
        )

        self.add_message("assistant", reply)
        return reply
