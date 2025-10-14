from typing import Dict, Any, List
from app.core.llm import LLMClient
from app.core.keywords import extract_keywords
from app.core.memgpt_client import MemGPTClient

# --- 템플릿: 레벨 가이드 + 대화 톤 지침 ---
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

SYSTEM_SUMMARY_QA = """너는 뉴스 토론 파트너다.
목표:
1) 입력 텍스트를 3~5문장으로 요약한다(과장/수식 최소, 사실 우선, 자연스러운 대화 톤).
2) 템플릿 레벨 가이드를 반영해 서로 다른 각도의 개방형 질문 3개를 제시한다.
3) 결과는 반드시 JSON으로만 출력한다.

출력 형식(JSON):
{
  "summary": "<3~5문장 요약>",
  "open_questions": ["<질문1>", "<질문2>", "<질문3>"]
}

조건:
- 유도/편향 금지, 공손하고 중립.
- 초보 사용자도 이해 가능한 단어 선택.
- JSON 외 불필요한 텍스트 출력 금지.
"""

def build_summary_qa_prompt(text: str, level: str) -> str:
    lvl = (level or "beginner").lower()
    guide = LEVEL_GUIDES.get(lvl, LEVEL_GUIDES["beginner"])
    return f"""{CONVERSATIONAL_STYLE}
{guide}
[작업]
- 아래 텍스트를 요약하고, 레벨 가이드를 반영한 개방형 질문 3개를 생성하라.
[텍스트]
{text.strip()}
"""

class SummarizeService:
    def __init__(self, llm: LLMClient, mem: MemGPTClient):
        self.llm = llm
        self.mem = mem

    def summarize(self, user_id: str, text: str, level: str = "beginner") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_SUMMARY_QA},
            {"role": "user", "content": build_summary_qa_prompt(text, level)}
        ]
        raw = self.llm.generate(messages, max_tokens=420, temperature=0.2)

        import json, re
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("LLM did not return JSON")
        data = json.loads(m.group(0))

        kws = extract_keywords(text, top_k=8)

        out = {
            "summary": (data.get("summary") or "").strip(),
            "open_questions": data.get("open_questions") or [],
            "keywords": kws
        }

        try:
            self.mem.write_event(user_id, "summary_created", {
                "len": len(text),
                "level": level,
                "has_questions": True
            })
        except Exception:
            pass

        return out

    def keywords(self, text: str, top_k: int = 8) -> List[str]:
        return extract_keywords(text, top_k=top_k)
