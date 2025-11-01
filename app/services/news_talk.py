from app.core.kanana_client import generator

SYSTEM_PROMPT = """너는 뉴스토론 파트너다.
뉴스 요약과 사용자의 의견을 기반으로
논리적이고 탐구적인 질문을 생성해야 해.
경제, 사회, 정치 뉴스 속 용어(예: 금리, 환율, 인플레이션, GDP 등)를 이해하고
그 의미나 파급효과를 고려해 대화를 발전시켜.
답변은 반드시 탐구형 질문 한 문장으로 끝내.
"""

async def make_question(summary: str, opinion: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

[뉴스 요약]
{summary}

[사용자 의견]
{opinion}

[질문 시작]
"""
    output = generator(prompt)[0]["generated_text"]
    if "[질문 시작]" in output:
        result = output.split("[질문 시작]")[-1].strip()
    else:
        result = output.strip()
    result = result.split("\n")[0].strip()
    return result
