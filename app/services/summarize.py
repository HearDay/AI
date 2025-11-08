from typing import Dict
from app.services.llm import LLMClient
from app.core.memgpt_client import MemGPTClient
from app.core.keywords import extract_keywords


CONVERSATIONAL_STYLE = """[대화 톤 지침]
- 자연스럽고 공손한 말투로, 사람처럼 한 문장으로 말하라.
- 인위적인 JSON, 목록, 포맷 표시는 금지.
- "~했습니다.", "~인데요.", "~어떨까요?" 등의 자연스러운 구어체 사용.
"""

LEVEL_GUIDES = {
    "beginner": """[레벨 가이드]
- 기본 사실 중심(누가, 언제, 어디서, 무엇을)
- 평가나 해석 없이 핵심 사실만 말하기
""",
    "intermediate": """[레벨 가이드]
- 원인과 영향 중심(왜, 누구에게 어떤 영향)
- 현상과 배경을 자연스럽게 연결
""",
    "advanced": """[레벨 가이드]
- 대안과 가치판단 중심(해결책, 트레이드오프)
- 논리적인 이유나 근거를 간결하게 덧붙이기
"""
}


SYSTEM_SUMMARY_QA = """너는 뉴스 토론 AI다.
주어진 내용을 2~3문장 이내로 자연스럽게 요약하고,
그에 이어 사용자의 수준(레벨 가이드)에 맞는 탐구형 질문을 한 문장으로 덧붙여라.
출력은 사람의 말투로 이어지는 자연스러운 대화 형태여야 한다.
예: "요약하자면 ..., 그럼 ... 어떻게 보시나요?"
JSON, 불릿포인트, 구분기호 없이 한 문단으로 말하라.
"""


def build_summary_prompt(text: str, level: str) -> str:
    lvl = (level or "beginner").lower()
    guide = LEVEL_GUIDES.get(lvl, LEVEL_GUIDES["beginner"])
    return f"""{CONVERSATIONAL_STYLE}
{guide}
[작업]
- 아래 뉴스 내용을 요약하고, 레벨 가이드에 맞게 탐구형 질문을 1개만 자연스럽게 이어 말하라.
[뉴스 내용]
{text.strip()}
"""


class SummarizeService:
    def __init__(self, llm: LLMClient, mem: MemGPTClient):
        self.llm = llm
        self.mem = mem

    def summarize(self, user_id: str, text: str, level: str = "beginner") -> Dict[str, str]:
        messages = [
            {"role": "system", "content": SYSTEM_SUMMARY_QA},
            {"role": "user", "content": build_summary_prompt(text, level)},
        ]

        try:
            reply = self.llm.generate(messages, max_tokens=280, temperature=0.6)
            reply = reply.strip()
        except Exception as e:
            print(f"[summarize] Error: {e}")
            reply = "요약 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

        # 키워드 추출 (별도)
        keywords = extract_keywords(text, top_k=5)

        # MemGPT 기록
        try:
            self.mem.write_event(user_id, "summary_created", {
                "len": len(text),
                "level": level,
                "keywords": keywords
            })
        except Exception:
            pass

        return {
            "dialogue": reply,
            "keywords": keywords
        }
