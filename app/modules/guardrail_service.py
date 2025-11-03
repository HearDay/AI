import re
from typing import Tuple

# 기본 금칙어 리스트 (추후 확장 가능)
BANNED_WORDS = ["욕설", "비속어", "개인정보", "신상"]

def content_filter(text: str) -> Tuple[bool, str]:
    """
    응답 내용에서 부적절한 단어 탐지
    반환: (is_safe, reason)
    """
    for word in BANNED_WORDS:
        if re.search(word, text):
            return False, f"⚠️ 부적절한 표현이 감지되었습니다: '{word}'"
    return True, ""

def relevance_check(user_message: str, llm_reply: str) -> bool:
    """
    사용자의 입력과 LLM 응답의 의미적 관련성 간단 검증
    """
    user_keywords = set(user_message.lower().split())
    reply_keywords = set(llm_reply.lower().split())
    overlap = len(user_keywords & reply_keywords)
    return overlap > 1  # 키워드가 1개 이상 겹치면 관련 있다고 판단
