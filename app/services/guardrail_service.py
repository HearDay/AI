import re

"""
응답 Guardrail (콘텐츠 필터)
- 위험/비속어 등은 내부 로그로만 기록
- 사용자 응답에는 절대 경고문구 노출하지 않음
- '개인정보 보호' 같은 정상 문장은 무시
"""

# 필터링 기준 키워드
BAD_KEYWORDS = [
    "욕설", "비속어", "음란", "모욕", "테러", "혐오",
    "살인", "자살", "폭력", "마약", "불법"
]

# 예외로 허용할 안전 맥락
SAFE_PHRASES = [
    "개인정보 보호", "정치적 중립", "정치적 공정성", "법적 책임",
    "표현의 자유", "개인정보 처리방침"
]

def content_filter(text: str):
    """
    입력 텍스트에서 위험 단어를 탐지하고,
    심각하지 않으면 그대로 통과시킴.
    """
    if not text:
        return True, text

    lowered = text.lower()

    # 예외 문맥은 그냥 통과
    for phrase in SAFE_PHRASES:
        if phrase in lowered:
            return True, text

    # 위험 키워드 탐지
    for bad in BAD_KEYWORDS:
        if re.search(bad, lowered):
            # 로그에는 남기지만 사용자에게 표시 안 함
            print(f"[Guardrail] 필터 감지됨: '{bad}' → '{text[:50]}...'")
            return False, text  # 응답 그대로 반환 (경고 안 붙임)

    return True, text
