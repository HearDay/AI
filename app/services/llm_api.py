import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

API_URL = "https://api.upstage.ai/v1/chat/completions"


def run_llm(messages, model="solar-mini-250422", max_tokens=400, temperature=0.7):
    """
    Upstage Solar Chat API 호출 함수
    - messages: [{"role": "user", "content": "..."}]
    - model: "solar-mini-250422" (Upstage 최신 소형모델)
    """
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("환경 변수 'UPSTAGE_API_KEY'가 설정되지 않았습니다. (.env.local 확인)")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["output"][0]["content"][0]["text"].strip()
    except Exception as e:
        print(f"[Upstage API Error] {e}\nResponse: {response.text if 'response' in locals() else 'No response'}")
        return "(응답 오류 — Upstage API 호출 실패)"
