from app.core.llm import LLMClient

# 전역 인스턴스 (서버 실행 시 1회 로드)
_llm_client = LLMClient()


def run_llm(messages, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """
    Kanana-2.1B 모델을 통해 응답을 생성하는 함수.
    Args:
        messages (List[Dict]): [{"role": "system"/"user"/"assistant", "content": str}, ...]
        max_tokens (int): 생성할 최대 토큰 수
        temperature (float): 생성 다양성 (0.0~1.0)

    Returns:
        str: 모델 응답 텍스트
    """
    return _llm_client.generate(messages, max_tokens=max_tokens, temperature=temperature)


def simple_prompt(prompt_text: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """
    단일 문자열 프롬프트 입력용 단축 함수.
    내부적으로 run_llm을 호출하여 system/user 구조 없이 바로 생성.
    """
    messages = [{"role": "user", "content": prompt_text}]
    return run_llm(messages, max_tokens=max_tokens, temperature=temperature)
