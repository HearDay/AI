import re
from typing import Tuple

#금칙어, 추후 확장 가능

BANNED_WORDS = ["욕설", "비속어", "개인정보", "신상"] 

def content_filter(text: str) -> Tuple[bool, str]:
    for word in BANNED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return False, f"⚠️ 부적절한 표현이 감지되었습니다: '{word}'"
    return True, ""

def relevance_check(user_message: str, llm_reply: str) -> Tuple[bool, str]:
    user_keywords = set(_okt.nouns(user_message))
    reply_keywords = set(_okt.nouns(llm_reply))
    if len(user_keywords & reply_keywords) < 2:
        return False, "의미적 관련성이 낮습니다."
    return True, ""
