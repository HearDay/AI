from app.modules.llm import run_llm
from app.prompt_templates import CONVERSATIONAL_STYLE, LEVEL_GUIDES


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
    Kanana 기반 질문 생성기 (open_question / followup 통합)
    mode: 'open_question' (토론 시작용) or 'followup' (응답 후속용)
    """
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES["beginner"])
    keywords = _extract_keywords(context)
    hint = ", ".join(keywords)

    # --- 프롬프트 분기 ---
    if mode in ("open", "open_question"):
        system_prompt = f"""너는 뉴스 토론 파트너다.
{CONVERSATIONAL_STYLE}
{guide}

[작업]
- {hint}와 관련된 자연스러운 개방형 질문을 한 문장으로 생성하라.
- 과장/명령 금지, "~어떨까요?" "~보시나요?" 형태 허용.
- 문장 앞뒤에 불필요한 부호나 번호를 붙이지 말 것.
"""
        user_prompt = f"[뉴스 요약] {context}"

    elif mode == "followup":
        system_prompt = f"""너는 뉴스 토론 파트너다.
{CONVERSATIONAL_STYLE}
{guide}

[작업 지침]
- 사용자의 발언을 읽고, 공감 한 문장 + {hint} 관련 후속 질문 한 문장.
- 전체 2문장 이하로 말하라.
- "~할까요?" "~어떨까요?" "~보시나요?" 형태 허용.
- 라벨, 구분자, 번호를 붙이지 말고 바로 말하라.
"""
        user_prompt = f"[대화 내용] {context}"

    else:
        raise ValueError("mode must be 'open_question' or 'followup'")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        reply = run_llm(messages, max_tokens=200, temperature=0.7)
        reply = reply.strip().split("\n")[0]
    except Exception as e:
        print(f"[question_generator] Error: {e}")
        reply = "좋은 시각이에요. 그럼 이 주제에서 가장 중요하다고 생각하는 부분은 무엇인가요?"

    return reply
