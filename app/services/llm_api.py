import os
import requests

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
API_URL = "https://api.upstage.ai/v1/chat/completions"

def run_llm(messages, model="solar-mini-250422", max_tokens=400, temperature=0.7):
    """
    Upstage Solar Chat API 호출 함수
    messages = [{"role": "user", "content": "..."}] 형식
    """
    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["output"][0]["content"][0]["text"].strip()
