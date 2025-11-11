from app.services.llm_api import run_llm


class HybridMemoryAgent:
    """
    간단한 대화 메모리 + Upstage LLM 결합형 에이전트
    인터페이스는 chat()만 제공해서 feedback/discussion에서 그대로 사용 가능
    """

    def __init__(self, name: str = "discussion_agent"):
        self.name = name
        # 전체 대화 히스토리 저장용
        # 예: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.history: list[dict] = []
        print(f"[HybridMemory] initialized: {name}")

    def add_message(self, role: str, content: str):
        """내부 히스토리에 메시지 추가"""
        self.history.append({"role": role, "content": content})

    def get_recent_messages(self, limit: int = 5):
        """최근 n턴만 잘라서 반환"""
        return self.history[-limit:]

    def chat(self, user_input: str, role: str = "user") -> str:
        """
        사용자 발화 → 메모리 저장 → Solar API 호출 → 응답도 메모리에 저장
        """
        # 1) 사용자 메시지 저장
        self.add_message(role, user_input)

        # 2) 최근 히스토리를 대화문맥으로 묶기
        context = self.get_recent_messages(limit=5)
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in context
        )

        # 3) Upstage Solar LLM 호출
        reply = run_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "너는 뉴스 기반 토론 AI 파트너다. "
                        "사용자의 의견에 공감해 주고, 부드럽게 생각을 확장할 수 있는 질문을 던져라."
                    ),
                },
                {
                    "role": "user",
                    "content": conversation,
                },
            ],
            max_tokens=300,
            temperature=0.7,
        )

        # 4) 응답도 메모리에 저장
        self.add_message("assistant", reply)

        return reply
