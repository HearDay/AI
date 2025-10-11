from transformers import pipeline

class KeywordExtractor:
    def __init__(self):
        # Zero-Shot Classification 파이프라인 로드
        # 한국어 성능이 좋은 klue/roberta-large 모델을 사용합니다.
        # 처음 실행 시 모델을 다운로드하므로 시간이 걸릴 수 있습니다.
        print("키워드 추출 모델을 로드합니다...")
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="klue/roberta-large"
        )
        print("모델 로드가 완료되었습니다.")

    def extract(self, text: str, candidate_keywords: list[str], top_k: int = 3) -> list[str]:
        """
        주어진 텍스트에서 후보 키워드 중 가장 관련성 높은 키워드를 추출합니다.
        
        :param text: 분석할 원본 텍스트
        :param candidate_keywords: 키워드 후보 목록
        :param top_k: 반환할 상위 키워드 개수
        :return: 관련성 높은 순으로 정렬된 키워드 리스트
        """
        if not text or not candidate_keywords:
            return []
            
        # 모델을 사용하여 분류 작업 수행
        result = self.classifier(text, candidate_keywords, multi_label=True)
        
        # 결과에서 상위 k개의 레이블(키워드)만 추출하여 반환
        top_keywords = result['labels'][:top_k]
        
        return top_keywords

# 싱글턴 패턴처럼 애플리케이션 전체에서 하나의 인스턴스만 사용하도록 생성
keyword_extractor = KeywordExtractor()