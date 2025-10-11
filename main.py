from fastapi import FastAPI
import torch

# FastAPI 앱 인스턴스 생성
app = FastAPI()

@app.get("/")
def read_root():
    # PyTorch가 잘 설치되었는지 확인
    torch_version = torch.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "message": "안녕하세요! FastAPI 서버가 정상적으로 동작합니다.",
        "pytorch_version": torch_version,
        "device": f"PyTorch는 현재 '{device}'에서 실행될 준비가 되었습니다."
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}