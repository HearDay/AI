from app.services.llm_api import run_llm
from app.core.prompt_templates import LEVEL_GUIDES

def generate_question(context: str, mode: str = "open_question", level: str = "beginner") -> str:
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES["beginner"])

    if mode == "open_question":
        system_prompt = f"""
        너는 뉴스 기반 첫 질문을 생성하는 AI다.
        보통은 질문 형식이지만, 상황에 따라 짧은 의견형 문장도 자연스럽게 생성해도 된다.
        문장은 1문장으로 자연스럽게 끝내라.
        문체는 항상 공손한 존댓말로 표현한다.
        반말, 명령조, 친구 말투는 절대 사용하지 않는다.
        난이도에 따라 질문의 깊이와 관점이 달라지도록 표현한다.

        {guide}
        """
        user_prompt = context

    else:  # followup
        system_prompt = f"""
        너는 사용자의 의견을 읽고 자연스럽게 반응하는 토론 파트너다.
        상황에 따라 짧은 의견 또는 질문을 1~2문장으로 생성해도 된다.
        반드시 질문일 필요는 없다.
        공감·요약·관찰 등 자연스러운 흐름을 따라라.
        문체는 항상 공손한 존댓말로 표현한다.
        반말, 명령조, 친구 말투는 절대 사용하지 않는다.
        난이도에 따라 질문의 깊이와 관점이 달라지도록 표현한다.
        {guide}
        """
        user_prompt = context

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    reply = run_llm(messages, max_tokens=150, temperature=0.7).strip()

    # ※ 더 이상 물음표 강제 추가 안 함
    # reply = reply.split("\n")[0].strip()
    # return reply

    # 여러 줄이면 첫 줄만 사용 (안정성)
    reply = reply.split("\n")[0].strip()
    return reply
