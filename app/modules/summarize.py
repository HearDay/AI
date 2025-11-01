from app.services.summarize import SummarizeService
from app.core.llm import LLMClient
from app.core.memgpt_client import MemGPTClient

# 인스턴스는 서버 시작 시 1회만 생성
_llm = LLMClient()
_mem = MemGPTClient()
_summarizer = SummarizeService(_llm, _mem)


def summarize_text(user_id: str, text: str, level: str = "beginner"):
    """
    뉴스 요약 + 질문 + 키워드 추출을 수행하는 헬퍼 함수.
    결과는 대화형 문자열(dialogue)과 keywords를 포함.
    """
    result = _summarizer.summarize(user_id, text, level)
    return result
