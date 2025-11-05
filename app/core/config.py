import os

class Settings:
    
    HF_MODEL_ID: str = os.getenv(
        "HF_MODEL_ID",
        "kakaocorp/kanana-1.5-2.1b-instruct-2505" 
    )
    HF_MAX_NEW_TOKENS: int = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
    HF_TEMPERATURE: float = float(os.getenv("HF_TEMPERATURE", "0.7"))
    HF_TOP_P: float = float(os.getenv("HF_TOP_P", "0.9"))
    DEVICE_MAP: str = os.getenv("HF_DEVICE_MAP", "auto")

settings = Settings()
