import re
from typing import Tuple

# 금칙어, 추후 확장 가능
BANNED_WORDS = ["욕설", "비속어", "개인정보", "신상"]

def content_filter(text: str) -> Tuple[bool, str]:
    """
    텍스트 내 부적절한 단어 감지
    """
    for word in BANNED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return False, f"⚠️ 부적절한 표현이 감지되었습니다: '{word}'"
    return True, ""

def relevance_check(user_message: str, llm_reply: str) -> Tuple[bool, str]:
    """
    사용자의 입력과 LLM 응답 간 의미적 관련성 검증 (단순 키워드 기반)
    """
    
    user_keywords = set(re.findall(r"\w+", user_message.lower()))
    reply_keywords = set(re.findall(r"\w+", llm_reply.lower()))

    overlap = len(user_keywords & reply_keywords)
    if overlap < 2:
        return False, "의미적 관련성이 낮습니다."
    return True, ""
