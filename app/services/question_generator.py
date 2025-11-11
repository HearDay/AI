from app.services.llm_api import run_llm  # 🔹 Upstage Solar API 버전
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
    """뉴스/토론 질문 생성기 (Solar API 기반)"""
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES["beginner"])
    keywords = _extract_keywords(context)
    hint = ", ".join(keywords) if keywords else "주제"

    # 모드 정규화
    if mode == "open":
        mode = "open_question"

    if mode == "open_question":
        system_prompt = f"""너는 '뉴스 토론 파트너' 역할의 AI다.
{CONVERSATIONAL_STYLE}
{guide}

[역할 요약]
- 사용자가 뉴스 요약을 기반으로 생각을 확장할 수 있도록 돕는다.
- 단정, 평가, 명령, 권유는 금지한다.
- 대화는 부드럽게, 자연스럽게 유도한다.

[출력 형식]
- 반드시 한 문장으로, 질문으로만 끝내라.
- "~어떨까요?", "~보시나요?", "~가능할까요?" 형태의 어미를 사용한다.
- 번호, 괄호, 따옴표, 메타표현 금지.

[작업 지시]
- 아래 뉴스 요약을 읽고 '{hint}'와 관련된 개방형 질문을 만들어라.
"""
        user_prompt = f"[뉴스 요약]\n{context}"

    elif mode == "followup":
        system_prompt = f"""너는 '뉴스 토론 파트너' 역할의 AI다.
{CONVERSATIONAL_STYLE}
{guide}

[역할 요약]
- 사용자의 발언을 읽고, 공감 + 논리 확장 질문을 제시한다.
- 공감은 부드럽고 간결하게, 질문은 관련 주제 내에서 이어지게 한다.

[출력 형식]
- 2문장 이내 (공감 1 + 질문 1)
- "~어떨까요?", "~생각하시나요?", "~가능할까요?" 형태로 마무리.
- "좋아요", "알겠습니다" 같은 패턴 금지.

[작업 지시]
- 사용자의 발언을 바탕으로 '{hint}'와 연관된 후속 질문을 만들어라.
"""
        user_prompt = f"[사용자 발언]\n{context}"

    else:
        raise ValueError("mode must be 'open_question' or 'followup'")

    # Upstage Solar API 메시지 형식
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        reply = run_llm(messages, max_tokens=250, temperature=0.65)
        reply = reply.strip().split("\n")[0].strip()
        reply = reply.strip(" \"'")

        if not reply.endswith(("?", "?!", "!?")):
            reply += "?"
        return reply

    except Exception as e:
        print(f"[question_generator] LLM call failed: {e}")
        return "좋은 생각이에요. 이 주제에서 특히 중요한 부분은 뭐라고 생각하시나요?"
