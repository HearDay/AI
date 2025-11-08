from transformers import pipeline
import asyncio  # 1. asyncio 임포트

class KeywordExtractor:
    def __init__(self):
        print("키워드 추출 모델을 로드합니다...")
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="pongjin/roberta_with_kornli"
        )
        print("모델 로드가 완료되었습니다.")

    # 2. 'def'를 'async def'로 변경
    async def extract(self, text: str, candidate_keywords: list[str], top_k: int = 3) -> list[str]:
        """
        [수정됨] CPU를 막는 classifier()를 별도 스레드에서 실행합니다.
        """
        if not text or not candidate_keywords:
            return []
            
        hypothesis_template = "이 텍스트는 {}에 관한 것입니다."
        
        # 3. '무거운' 동기 작업(classifier)을 별도 스레드에서 실행
        result = await asyncio.to_thread(
            self.classifier,
            text, 
            candidate_keywords, 
            hypothesis_template=hypothesis_template, 
            multi_label=True
        )
        
        top_keywords = result['labels'][:top_k]
        
        return top_keywords

# 싱글턴 인스턴스 생성
keyword_extractor = KeywordExtractor()