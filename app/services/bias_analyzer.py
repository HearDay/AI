from transformers import pipeline
import torch

class BiasAnalyzer:
    def __init__(self):
        print("편향성 분석 모델을 로드합니다...")
        # 한국어 NLI 모델 사용
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="pongjin/roberta_with_kornli",
            device=0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
        )
        print("편향성 분석 모델 로드 완료.")

    # 👇 [수정됨] async def -> def (동기 함수로 변경)
    def analyze_bias(self, text: str) -> dict:
        """
        기사 본문을 분석하여 'NEUTRAL'(중립) 또는 'BIASED'(편향) 라벨을 반환합니다.
        
        [긴급 수정] 필터링 기준 대폭 완화
        - 뉴스 기사가 과도하게 필터링되는 것을 막기 위해 '편향' 판정 기준을 엄격하게 높입니다.
        """
        if not text:
            return {"label": "UNKNOWN", "score": 0.0}

        # 1. 라벨을 '뉴스' vs '개인 의견'으로 변경하여 일반 기사가 걸러지는 것 방지
        label_neutral = "사실을 전달하는 뉴스 보도"
        label_biased = "글쓴이의 주관적인 주장이 강한 글"
        
        candidate_labels = [label_neutral, label_biased]
        
        # 가설 템플릿
        hypothesis_template = "이 글은 {}입니다."

        short_text = text[:512]

        # AI 추론 실행 (await 없이 바로 실행)
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

        # 점수 매핑
        scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        score_biased = scores.get(label_biased, 0.0)
        score_neutral = scores.get(label_neutral, 0.0)

        print(f"[Bias Debug] 중립({score_neutral:.4f}) vs 편향({score_biased:.4f})")

        # 2. 임계값(Threshold)을 0.85로 상향 (확실한 것만 거름)
        BIAS_THRESHOLD = 0.85

        if score_biased >= BIAS_THRESHOLD:
            # 편향 점수가 압도적으로 높을 때만 BIASED 리턴
            return {"label": "BIASED", "score": score_biased}
        else:
            # 그 외에는 모두 NEUTRAL (안전하게 통과)
            final_score = score_neutral if score_neutral > score_biased else (1.0 - score_biased)
            return {"label": "NEUTRAL", "score": final_score}

# 싱글턴 인스턴스 생성
bias_analyzer = BiasAnalyzer()