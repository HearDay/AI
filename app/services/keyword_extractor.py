from transformers import pipeline

class KeywordExtractor:
    """뉴스 기사에서 키워드를 추출하는 서비스"""
    
    def __init__(self):
        print("키워드 추출 모델을 로드합니다...")
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="pongjin/roberta_with_kornli"
        )
        print("모델 로드가 완료되었습니다.")

    def extract(self, text: str, candidate_keywords: list[str], top_k: int = 3) -> list[str]:
        """
        텍스트에서 후보 키워드 중 가장 관련성 높은 키워드를 추출합니다.
        
        Args:
            text: 분석할 텍스트
            candidate_keywords: 후보 키워드 리스트
            top_k: 반환할 상위 키워드 개수
            
        Returns:
            list[str]: 추출된 키워드 리스트
        """
        if not text or not candidate_keywords:
            return []
            
        hypothesis_template = "이 텍스트는 {}에 관한 것입니다."
        
        result = self.classifier(
            text, 
            candidate_keywords, 
            hypothesis_template=hypothesis_template, 
            multi_label=True
        )
        
        return result['labels'][:top_k]

keyword_extractor = KeywordExtractor()