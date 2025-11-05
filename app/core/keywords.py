from typing import List
import re

STOPWORDS = set("""
가 나 다 라 마 바 사 아 자 차 카 타 파 하 그리고 그러나 하지만 그래서 또는 혹은 또한 매우 너무
이 그 저 등 등등 또한 이미 또한 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열 을 를 이 가 은 는 의 에 에서 으로 로 와 과 도 에게
에서부터 까지 보다 라고 라는 등이 라든가 랑 하고 또는 및 등등 대한 대해 대해서 위해 위한 인한
""".split())

def extract_keywords(text: str, top_k: int = 8, min_len: int = 2) -> List[str]:
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", text).lower()
    toks = [w for w in t.split() if len(w) >= min_len and w not in STOPWORDS]
    if not toks:
        return []
    freq = {}
    N = len(toks)
    for i, w in enumerate(toks):
        score = 1.0 + (1.0 - (i / max(1, N)))  # 앞쪽 가중
        freq[w] = freq.get(w, 0.0) + score
    return [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]