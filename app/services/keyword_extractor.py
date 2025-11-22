from transformers import pipeline
import asyncio
from fastapi.concurrency import run_in_threadpool  # 1. run_in_threadpool 임포트

class KeywordExtractor:
    def __init__(self):
        print("키워드 추출 모델을 로드합니다...")
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="pongjin/roberta_with_kornli"
        )
        print("모델 로드가 완료되었습니다.")

    def extract(self, text: str, candidate_keywords: list[str], top_k: int = 3) -> list[str]:
        """
        [수정됨] CPU를 막는 classifier()를 FastAPI의 스레드풀에서 실행합니다.
        """

        if not text or not candidate_keywords:
            return []
            
        hypothesis_template = "이 텍스트는 {}에 관한 것입니다."
        
        # 2. '무거운' 동기 작업(classifier)을 run_in_threadpool로 실행
        result = self.classifier(
            text, 
            candidate_keywords, 
            hypothesis_template=hypothesis_template, 
            multi_label=True
        )
        
        top_keywords = result['labels'][:top_k]
        
        return top_keywords

keyword_extractor = KeywordExtractor()