import re
from app.services.llm_api import run_llm
from app.core.prompt_templates import LEVEL_GUIDES

def _one(text, max_s=2):
    text = text.replace("\n", " ").strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(s.strip() for s in sents[:max_s] if s.strip())


def generate_question(context, mode="followup", level="beginner"):
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES["beginner"])

    # 요약 요청
    if any(k in context for k in ["요약", "정리", "간단히 설명"]):
        sp = (
            "다음 내용을 2~3문장으로 공손하게 요약해라. "
            "예시는 필요하다고 판단되는 경우에만 생성할 수 있다."
        )
        res = run_llm(
            [
                {"role": "system", "content": sp},
                {"role": "user", "content": context}
            ],
            max_tokens=120, temperature=0.4
        )
        return _one(res, 3)

    # open question
    if mode == "open_question":
        sp = (
            "뉴스 요약을 읽고 자연스러운 첫 질문을 1~2문장으로 생성해라. "
            "예시는 필요하다고 판단될 때만 포함해라.\n"
            f"{guide}"
        )
    else:
        sp = (
            "사용자의 의견을 읽고 follow-up 질문 또는 반응을 1~2문장으로 생성해라. "
            "질문은 0~1개 포함할 수 있으며 예시는 필요할 때만 포함한다.\n"
            f"{guide}"
        )

    raw = run_llm(
        [
            {"role": "system", "content": sp},
            {"role": "user", "content": context}
        ],
        max_tokens=120, temperature=0.5
    )

    return _one(raw, 2)
