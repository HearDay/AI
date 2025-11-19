from app.services.llm_api import run_llm

class HybridMemoryAgent:
    """
    간단한 대화 메모리 + Upstage LLM 결합형 에이전트 (개선 버전)
    - followup 응답이 훨씬 자연스럽고 풍부하도록 system_prompt 확장
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
        """
        followup 모드 기본 의견 생성 담당
        - 질문 금지
        - 자연스러운 1~2문장 의견/관찰/부연 설명 허용
        - 과도한 딱딱함 제거
        """
        self.add_message(role, user_input)

        context = self.get_recent_messages(limit=5)
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in context)

        reply = run_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "너는 사용자의 발화를 듣고 자연스럽고 부드러운 방식으로 의견을 말하는 토론 파트너다. "
                        "1~2문장으로 대화하듯 자연스럽게 반응하되, 문체는 항상 공손한 **존댓말**을 사용한다. "
                        "공감, 간단한 설명, 관찰, 의견 표현 모두 가능하다. "
                        "지나치게 딱딱한 분석은 피하고, 사람처럼 자연스럽게 말해라. "
                        "단, 질문은 생성하지 마라."
                    ),
                },
                {"role": "user", "content": conversation},
            ],
            max_tokens=150,
            temperature=0.7,
        )

        self.add_message("assistant", reply)
        return reply
