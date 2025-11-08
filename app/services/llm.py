import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


class LLMClient:
    """Kanana-1.5-2.1B-Instruct-2505 로컬 모델 직접 호출"""

    def __init__(self):
        self.model_id = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Device set to use {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.7,
    ) -> str:
        """Chat-like 메시지를 기반으로 텍스트 생성"""
        
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

        # 토큰화 & 추론
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,  # ✅ 명시적으로 EOS 설정
            )

        # 디코딩
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        #
        output_text = output_text.replace("<|user|>", "").strip()
        if "<|assistant|>" in output_text:
            output_text = output_text.split("<|assistant|>")[-1].strip()

        return output_text.strip()


_llm_client = LLMClient()


def run_llm(messages, max_tokens: int = 400, temperature: float = 0.7) -> str:
    """LLMClient 기반 메시지 생성"""
    return _llm_client.generate(messages, max_tokens=max_tokens, temperature=temperature)


def simple_prompt(prompt_text: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
    """단일 텍스트 입력용 헬퍼"""
    return run_llm(
        [{"role": "user", "content": prompt_text}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
