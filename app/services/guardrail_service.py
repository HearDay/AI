import re
from typing import Tuple

# ------------------------------------------------------------
# 콘텐츠 안전성 필터 (Guardrail)
# ------------------------------------------------------------

# 금칙어 목록 — 필요 시 확장 가능
BANNED_WORDS = ["욕설", "비속어", "개인정보", "신상"]


def content_filter(text: str) -> Tuple[bool, str]:
    """
    텍스트 내 부적절한 단어 감지
    - 금칙어 포함 시 False 반환
    - 한글 경계도 정확히 감지
    """
    for word in BANNED_WORDS:
        # 한글 단어 경계까지 포함하도록 정규식 개선
        if re.search(rf"(?<![가-힣]){re.escape(word)}(?![가-힣])", text):
            return False, f"⚠️ 부적절한 표현이 감지되었습니다: '{word}'"
    return True, ""


def relevance_check(user_message: str, llm_reply: str) -> Tuple[bool, str]:
    """
    사용자의 입력과 LLM 응답 간 의미적 관련성 검증 (단순 키워드 기반)
    - 교집합 키워드가 2개 미만일 경우 경고
    """
    user_keywords = set(re.findall(r"\w+", user_message.lower()))
    reply_keywords = set(re.findall(r"\w+", llm_reply.lower()))

    overlap = len(user_keywords & reply_keywords)
    if overlap < 2:
        return False, "의미적 관련성이 낮습니다."
    return True, ""
