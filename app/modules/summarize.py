from typing import Dict
from app.core.llm import LLMClient
from app.core.memgpt_client import MemGPTClient
from app.core.keywords import extract_keywords

LEVEL_GUIDES = {
    "beginner": "- 기본 사실 중심",
    "intermediate": "- 원인과 영향 중심",
    "advanced": "- 가치 판단 중심",
}

class SummarizeService:
    def __init__(self, llm: LLMClient, mem: MemGPTClient):
        self.llm = llm
        self.mem = mem

    def summarize(self, user_id: str, text: str, level: str = "beginner") -> Dict[str, str]:
        prompt = f"""
너는 뉴스 요약 전문가다.
{text}
위 내용을 2~3문장으로 요약하고 {LEVEL_GUIDES[level]} 탐구형 질문을 한 문장 덧붙여라.
"""
        try:
            reply = self.llm.generate([{"role": "user", "content": prompt}], max_tokens=280, temperature=0.6).strip()
        except Exception:
            reply = "요약 중 오류가 발생했습니다."

        keywords = extract_keywords(text, top_k=5)
        try:
            self.mem.write_event(user_id, "summary_created", {"level": level, "keywords": keywords})
        except Exception:
            pass

        return {"dialogue": reply, "keywords": keywords}
