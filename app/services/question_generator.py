from app.services.llm import run_llm
from app.core.prompt_templates import CONVERSATIONAL_STYLE, LEVEL_GUIDES


def _extract_keywords(text: str, top_k: int = 5):
    """간단한 핵심 단어 추출"""
    words = [w.strip(",.!?") for w in text.split() if len(w) > 1]
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
        if len(seen) >= top_k:
            break
    return seen


def generate_question(context: str, mode: str = "open_question", level: str = "beginner") -> str:
    """
    Kanana 기반 뉴스 토론 질문 생성기 (open_question / followup 통합)
    mode: 'open_question' (토론 시작용) or 'followup' (응답 후속용)
    """
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES["beginner"])
    keywords = _extract_keywords(context)

    if not isinstance(keywords, (list, tuple)):
        keywords = [str(keywords)] if keywords is not None else []

    hint = ", ".join(keywords) if keywords else "주제"

    # ==========================
    # 🔹 뉴스 기반 탐구형 질문 (open_question)
    # ==========================
    if mode in ("open", "open_question"):
        system_prompt = f"""너는 '뉴스 토론 파트너' 역할을 맡은 AI다.
{CONVERSATIONAL_STYLE}
{guide}

[역할 요약]
- 너는 사용자와 함께 뉴스를 기반으로 사고를 확장하고 관점을 넓히는 대화 파트너다.
- 질문은 친근하면서도 깊이 있는 사고를 유도해야 한다.
- 단정, 평가, 명령, 권유는 금지한다.

[출력 형식]
- 반드시 한 문장으로, 질문으로만 끝내라.
- "~어떨까요?", "~보시나요?", "~가능할까요?" 등 탐구형 어미를 사용한다.
- 문장 앞뒤에 번호, 기호, 괄호, 따옴표를 절대 붙이지 마라.

[작업 지시]
- 아래 요약된 뉴스 내용을 기반으로, '{hint}'와 관련된 개방형 질문을 만들어라.
- 너무 넓거나 모호한 질문 대신, 사용자가 생각을 구체적으로 표현할 수 있는 수준으로 설계하라.
"""
        user_prompt = f"[뉴스 요약]\n{context}"

    # ==========================
    # 🔹 Follow-up (후속 질문)
    # ==========================
    elif mode == "followup":
        system_prompt = f"""너는 '뉴스 토론 파트너' 역할을 맡은 AI다.
{CONVERSATIONAL_STYLE}
{guide}

[역할 요약]
- 사용자의 발언을 읽고 감정적 공감을 표현한 뒤, 같은 주제 내에서 사고를 확장하는 질문을 제시한다.
- 공감 표현은 부드럽고 간결하게, 질문은 논리적으로 연결되게 작성한다.

[출력 형식]
- 전체 2문장 이내 (공감 1문장 + 질문 1문장)
- "~어떨까요?", "~생각하시나요?", "~가능할까요?" 형태로 끝맺는다.
- 절대 "알겠습니다", "좋습니다" 같은 패턴 문장은 쓰지 마라.
- 번호, 따옴표, 구분자, 메타표현은 절대 포함하지 않는다.

[작업 지시]
- 사용자의 발언을 바탕으로 '{hint}'와 연관된 후속 질문을 생성하라.
- 공감은 발언의 감정적 흐름(찬성, 우려, 중립 등)에 맞게 자연스럽게 표현하라.
"""
        user_prompt = f"[사용자 발언]\n{context}"

    else:
        raise ValueError("mode must be 'open_question' or 'followup'")

    # ==========================
    # 🔹 메시지 구성 및 실행
    # ==========================
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        reply = run_llm(messages, max_tokens=250, temperature=0.65)
        # 여러 줄 생성 시 첫 줄만 사용
        reply = reply.strip().split("\n")[0].strip()
        # 안전 필터링
        if not reply.endswith("?"):
            reply += "?"
    except Exception as e:
        print(f"[question_generator] Error: {e}")
        reply = "좋은 시각이에요. 그럼 이 주제에서 가장 중요하다고 생각하는 부분은 무엇인가요?"

    return reply
