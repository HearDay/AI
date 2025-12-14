from transformers import pipeline
import torch

class BiasAnalyzer:
    """뉴스 기사 편향성 분석 서비스"""
    
    BIAS_THRESHOLD = 0.85  # 편향 판정 임계값
    
    def __init__(self):
        print("편향성 분석 모델을 로드합니다...")
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="pongjin/roberta_with_kornli",
            device=0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
        )
        print("편향성 분석 모델 로드 완료.")

    def analyze_bias(self, text: str) -> dict:
        """
        기사 본문을 분석하여 편향성 여부를 판단합니다.
        
        Returns:
            dict: {"label": "NEUTRAL" | "BIASED" | "UNKNOWN", "score": float}
        """
        if not text:
            return {"label": "UNKNOWN", "score": 0.0}

        label_neutral = "사실을 전달하는 뉴스 보도"
        label_biased = "글쓴이의 주관적인 주장이 강한 글"
        candidate_labels = [label_neutral, label_biased]
        hypothesis_template = "이 글은 {}입니다."
        short_text = text[:512]

        try:
            result = self.classifier(
                short_text,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=False
            )
        except Exception as e:
            print(f"[BiasAnalyzer Error] 분석 실패, 기본값(NEUTRAL) 반환: {e}")
            return {"label": "NEUTRAL", "score": 0.0}

        scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        score_biased = scores.get(label_biased, 0.0)
        score_neutral = scores.get(label_neutral, 0.0)

        print(f"[Bias Debug] 중립({score_neutral:.4f}) vs 편향({score_biased:.4f})")

        if score_biased >= self.BIAS_THRESHOLD:
            return {"label": "BIASED", "score": score_biased}
        else:
            final_score = score_neutral if score_neutral > score_biased else (1.0 - score_biased)
            return {"label": "NEUTRAL", "score": final_score}

# 싱글턴 인스턴스 생성
bias_analyzer = BiasAnalyzer()