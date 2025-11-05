import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict


class LLMClient:
    """Kanana-1.5-2.1B-Instruct-2505 로컬 모델을 직접 호출하는 LLMClient"""

    def __init__(self):
        self.model_id = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        # pipeline 초기화
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            repetition_penalty=1.1,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        """
        OpenAI Chat 형식(messages=[{role, content}, ...])을 받아
        Kanana 모델에서 직접 추론.
        """
        # system/user/assistant 구조를 단일 프롬프트로 합침
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"[시스템 지시]\n{content}\n\n"
            elif role == "user":
                prompt += f"[사용자]\n{content}\n\n"
            elif role == "assistant":
                prompt += f"[AI 응답]\n{content}\n\n"

        # 실제 추론
        output = self.generator(prompt, max_new_tokens=max_tokens, temperature=temperature)[0]["generated_text"]

        # 프롬프트 이후의 생성문만 추출
        if prompt in output:
            output = output.split(prompt)[-1].strip()

        return output.strip()



_llm_client = LLMClient()

def run_llm(messages, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """LLMClient 기반 메시지 생성"""
    return _llm_client.generate(messages, max_tokens=max_tokens, temperature=temperature)

def simple_prompt(prompt_text: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """단일 텍스트 입력용 헬퍼"""
    return run_llm([{"role": "user", "content": prompt_text}], max_tokens=max_tokens, temperature=temperature)
