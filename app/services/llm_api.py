import os
import requests

# Upstage Solar Chat API 기본 설정
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
API_URL = "https://api.upstage.ai/v1/solar/chat/completions"

def run_llm(messages, model="solar-mini-250422", max_tokens=400, temperature=0.7):
    """
    Upstage Solar Chat API 호출 함수
    - messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    if not UPSTAGE_API_KEY:
        raise ValueError("환경변수 UPSTAGE_API_KEY가 설정되지 않았습니다.")

    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "(응답 오류 — Upstage API 호출 실패)"
