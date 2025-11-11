import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import asyncio

class LLMClient:
    """H2O-Danube-1.8B Chat 모델 (Apache-2.0)"""

    def __init__(self):
        # 모델 ID 및 장치 설정
        self.model_id = "h2oai/h2o-danube2-1.8b-chat"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LLM] Device set to use {self.device}")

        # 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.7,
    ) -> str:
        """대화형 메시지 기반 텍스트 생성"""
        # system + user 메시지 합침
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"

        # 토큰화 및 모델 추론
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 결과 디코딩
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # <|assistant|> 이후 텍스트만 남김 (간혹 여러 구간 존재 시 마지막 사용)
        if "<|assistant|>" in output_text:
            output_text = output_text.split("<|assistant|>")[-1]
        output_text = output_text.replace("<|user|>", "").strip()

        return output_text.strip()


# -------------------------------------------------------
# 전역 클라이언트 (앱 시작 시 1회 로드)
# -------------------------------------------------------
_llm_client = None

async def get_llm_client() -> LLMClient:
    """전역 LLM 인스턴스 비동기 초기화"""
    global _llm_client
    if _llm_client is None:
        _llm_client = await asyncio.to_thread(LLMClient)
    return _llm_client


# -------------------------------------------------------
# 헬퍼 함수
# -------------------------------------------------------
def run_llm(messages, max_tokens: int = 400, temperature: float = 0.7) -> str:
    """LLMClient 기반 메시지 생성"""
    client = _llm_client or LLMClient()
    return client.generate(messages, max_tokens=max_tokens, temperature=temperature)


def simple_prompt(prompt_text: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
    """단일 텍스트 입력용 헬퍼"""
    return run_llm(
        [{"role": "user", "content": prompt_text}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
