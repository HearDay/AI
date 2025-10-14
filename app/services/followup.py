from typing import Dict, Any
from app.core.llm import LLMClient
from app.core.memgpt_client import MemGPTClient

CONVERSATIONAL_STYLE = """[대화 톤 지침]
- 사람처럼 자연스럽게, 과장/명령조 금지.
- 간단한 추임새 허용: "음,", "좋은 포인트예요.", "그렇게 볼 수도 있겠네요."
- 1~2문장 단락, 장황한 수식어 금지, 공손하고 중립.
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

# followup 전용 시스템 프롬프트
SYSTEM_FOLLOWUP = """너는 뉴스 토론 '지식 성장 파트너'다.
목표:
1) 사용자의 응답 품질을 간결히 평가한다(강점 1~2, 보완점 1~2, 각 80자 이내).
2) 보완점에 즉시 적용 가능한 '정중한' 팁 1개를 제시한다(명령조 금지).
3) 레벨 가이드와 대화 톤 지침을 따르는 '개방형 후속 질문' 1개를 생성한다.

형식(JSON만 출력):
{"score": <0~100 정수>, "strengths": ["..."], "improvements": ["..."], "tip": "...", "followup": "..."}

출력 조건:
- JSON 외 불필요한 문구 금지.
- 'tip'은 명령형 대신 권유형 표현 사용(예: "~해 보시면 좋겠습니다").
- 'followup'은 한 문장, 유도/편향 금지, 자연스러운 대화 톤 유지.
"""

def build_user_prompt(question: str, user_answer: str, rubric: str, level: str) -> str:
    level_key = (level or "beginner").lower()
    guide = LEVEL_GUIDES.get(level_key, LEVEL_GUIDES["beginner"])
    rubric_text = rubric if rubric else "기본: 명확성, 근거, 구조성, 맥락 적합성"

    return f"""{CONVERSATIONAL_STYLE}
{guide}
[원문 질문]
{question}

[사용자 응답]
{user_answer}

[평가 기준]
{rubric_text}

[작업]
- 위 응답을 평가하고, 레벨 가이드와 대화 톤 지침을 반영해 '후속 질문' 1개를 생성하라.
- 반드시 지정된 JSON 형식으로만 출력한다.
"""

class FollowupService:
    def __init__(self, llm: LLMClient, mem: MemGPTClient):
        self.llm = llm
        self.mem = mem

    def run(self, user_id: str, question: str, user_answer: str, rubric: str = "", level: str = "beginner") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_FOLLOWUP},
            {"role": "user", "content": build_user_prompt(question, user_answer, rubric, level)}
        ]
        text = self.llm.generate(messages, max_tokens=340, temperature=0.2)

        # LLM은 반드시 JSON을 반환해야 함. 아니면 예외 발생.
        import json, re
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("LLM did not return JSON")
        data = json.loads(m.group(0))

        try:
            self.mem.write_event(user_id, "feedback_generated", {
                "question": question,
                "level": level,
                "summary": data
            })
        except Exception:
            pass
        return data
