"""
레벨별(초급/중급/심화) 탐구형 질문 템플릿 + 자연스러운 대화 톤 지침.
- beginner : 사실 이해
- intermediate : 원인/영향
- advanced : 대안/가치판단
"""

LEVEL_GUIDES = {
    "beginner": """[레벨 가이드]
- 사실 이해 중심(누가, 언제, 어디서, 무엇을)
- 문장은 짧고 명료하게
- 판단을 보류하고 중립적으로 표현
""",
    "intermediate": """[레벨 가이드]
- 원인/영향 중심(왜, 누구에게 어떤 영향)
- 단기/장기 관점 구분
- 반례를 조심스럽게 제시
""",
    "advanced": """[레벨 가이드]
- 대안/가치판단 중심(해결책, 원칙, 트레이드오프)
- 수용 가능한 타협안과 리스크 제시
- 근거 유형(통계, 전문가, 사례) 포함
"""
}

CONVERSATIONAL_STYLE = """[대화 톤 지침]
- 사람처럼 자연스럽게, 과장·명령조 금지.
- 간단한 추임새 허용: "음,", "그럴 수도 있겠네요.", "좋은 생각이에요."
- 1~2문장 단락, 공손하고 중립적인 어조 유지.
- 문장은 질문 또는 대화로 끝나도록 한다.
"""

def build_open_question_prompt(summary: str, level: str = "beginner") -> str:
    """
    뉴스 요약문을 받아서 레벨별 탐구형 질문을 한 문장으로 생성하는 기본 프롬프트.
    """
    lvl = (level or "beginner").lower()
    guide = LEVEL_GUIDES.get(lvl, LEVEL_GUIDES["beginner"])

    return f"""너는 뉴스 토론 파트너다.
{CONVERSATIONAL_STYLE}
{guide}

[작업]
- 아래 뉴스 요약을 읽고, 사용자와 대화하듯 한 문장의 개방형 탐구 질문을 만들어라.
- '~어떨까요?', '~보시나요?', '~생각하시나요?' 같은 부드러운 형태 허용.
- 유도·편향 금지, 초보 사용자도 이해할 수 있는 표현 사용.

[뉴스 요약]
{summary.strip()}
"""

def conversational_rewrite(text: str) -> str:
    """
    모델이 다소 딱딱한 어투로 답했을 때, 자연스럽게 다듬기 위한 후처리.
    """
    # 형식적 명령문 → 제안형 문장으로 변환
    text = (
        text.replace("해야 한다", "하면 좋겠습니다")
            .replace("하라", "해 볼까요")
            .replace("하십시오", "해 볼까요")
            .replace("것이다", "것 같아요")
    )
    # 불필요한 기호, 공백 제거
    lines = [l.strip(" -•") for l in text.splitlines() if l.strip()]
    return " ".join(lines)
