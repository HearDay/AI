from app.core.llm import LLMClient

_llm_client = LLMClient()

def run_llm(messages, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """LLMClient 기반 메시지 생성"""
    return _llm_client.generate(messages, max_tokens=max_tokens, temperature=temperature)

def simple_prompt(prompt_text: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """단일 텍스트 입력용 헬퍼"""
    return run_llm([{"role": "user", "content": prompt_text}], max_tokens=max_tokens, temperature=temperature)
