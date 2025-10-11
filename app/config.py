import os

class Settings:
    HF_MODEL_ID: str = os.getenv(
        "HF_MODEL_ID",
        "yanolja/YanoljaNEXT-EEVE-Instruct-7B-v2-Preview"
    )
    HF_MAX_NEW_TOKENS: int = int(os.getenv("HF_MAX_NEW_TOKENS", "320"))
    HF_TEMPERATURE: float = float(os.getenv("HF_TEMPERATURE", "0.7"))
    HF_TOP_P: float = float(os.getenv("HF_TOP_P", "0.9"))
    DEVICE_MAP: str = os.getenv("HF_DEVICE_MAP", "auto")

settings = Settings()
