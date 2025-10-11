from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# 2단계에서 만든 서비스 모듈을 가져옵니다.
from .services.keyword_extractor import keyword_extractor

app = FastAPI()

# Pydantic 모델을 사용하여 요청 본문의 데이터 형식을 정의합니다.
class KeywordRequest(BaseModel):
    text: str = Field(..., min_length=1, description="분석할 텍스트")
    candidates: List[str] = Field(..., min_items=1, description="키워드 후보 목록")

class KeywordResponse(BaseModel):
    keywords: List[str]

@app.get("/")
def read_root():
    return {"message": "LLM을 활용한 키워드 추출 API"}

@app.post("/extract-keywords", response_model=KeywordResponse)
def extract_keywords_endpoint(request: KeywordRequest):
    """
    입력된 텍스트에서 핵심 키워드를 추출합니다.
    """
    # 서비스의 extract 함수를 호출하여 결과를 받습니다.
    extracted_keywords = keyword_extractor.extract(
        text=request.text, 
        candidate_keywords=request.candidates
    )
    
    return {"keywords": extracted_keywords}