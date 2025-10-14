import os
import requests
from typing import List, Dict

class LLMClient:
    """
    1차 LLM만 사용. 실패/타임아웃 시 예외를 그대로 던진다(폴백 없음).
    환경변수:
      - PRIMARY_LLM_URL: LLM 엔드포인트 (필수)
      - LLM_TIMEOUT: 요청 타임아웃(초), 기본 8
    """

    def __init__(self):
        self.primary_url = os.getenv("PRIMARY_LLM_URL", "")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "8"))

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.3) -> str:
        if not self.primary_url:
            raise RuntimeError("PRIMARY_LLM_URL not set")

        try:
            resp = requests.post(
                self.primary_url,
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # 엔드포인트 스펙에 따라 아래 두 가지를 지원
            return data.get("text") or data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"LLM primary failed: {e}") from e
