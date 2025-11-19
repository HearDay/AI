from app.services.llm_api import run_llm
from app.core.prompt_templates import LEVEL_GUIDES

def generate_question(context: str, mode: str = "open_question", level: str = "beginner") -> str:
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES["beginner"])

    if mode == "open_question":
        system_prompt = f"""
        너는 뉴스 기반 첫 질문을 생성하는 AI다.
        반드시 1문장으로 자연스럽게 끝내라.
        {guide}
        """
        user_prompt = context

    else:  # followup
        system_prompt = f"""
        너는 사용자의 의견을 읽고 자연스러운 후속 질문을 생성하는 토론 파트너다.
        중립적이고 부드러운 어조로, 상황에 따라 1~2문장으로 생성해도 된다.
        공감이나 설명을 넣지 말고, 질문만 생성하라.
        답변은 부드러운 개방형 질문으로 끝내라.
        {guide}
        """
        user_prompt = context

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    reply = run_llm(messages, max_tokens=120, temperature=0.65).strip()
    reply = reply.split("\n")[0].strip()

    if not reply.endswith("?"):
        reply += "?"

    return reply
