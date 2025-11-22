from sqlalchemy.orm import Session
from sqlalchemy import select
import numpy as np
import json
from datetime import datetime, timedelta

# [수정됨] ArticleCluster 임포트 제거 (ArticleRecommend, ArticleRecommendVector만 사용)
from app.models.document import ArticleRecommend, ArticleRecommendVector

class ClusteringService:
    def __init__(self):
        # 유사도 임계값 (0.85 이상이면 같은 사건으로 간주)
        self.SIMILARITY_THRESHOLD = 0.85

    def assign_to_cluster(self, db: Session, reco_id: int, current_vector: list, title: str):
        """
        [동기] 편향된 기사를 기존 클러스터 ID에 할당하거나, 
        유사한 그룹이 없으면 자신의 ID를 새로운 그룹 ID로 부여합니다.
        """
        print(f"[클러스터링] ID {reco_id}: 그룹화 시작...")

        # 1. 비교 대상 조회: 최근 24시간 내의 'FILTERED' (편향된) 기사들 중 이미 그룹이 있는 기사들
        one_day_ago = datetime.now() - timedelta(days=1)
        
        query = (
            select(ArticleRecommend.id, ArticleRecommendVector.sbert_vector, ArticleRecommend.article_cluster_id)
            .join(ArticleRecommendVector, ArticleRecommend.id == ArticleRecommendVector.article_recommend_id)
            .where(ArticleRecommend.status == 'FILTERED')       # 편향된 기사만 비교
            .where(ArticleRecommend.id != reco_id)              # 자기 자신 제외
            .where(ArticleRecommend.created_at >= one_day_ago)  # 최근 기사만
            .where(ArticleRecommend.article_cluster_id != None) # 이미 그룹이 있는 기사만 비교 대상
        )
        
        rows = db.execute(query).all()
        
        best_similarity = -1.0
        target_cluster_id = None
        
        # 현재 기사 벡터 (numpy 변환)
        curr_vec = np.array(current_vector, dtype='float32')
        curr_norm = np.linalg.norm(curr_vec)

        # 2. 가장 유사한 기사 찾기 (Brute-force 비교)
        for other_reco_id, other_vector_json, cluster_id in rows:
            if not other_vector_json: continue
            
            # JSON 파싱 (혹시 문자열로 저장된 경우)
            if isinstance(other_vector_json, str):
                try: other_vec_list = json.loads(other_vector_json)
                except: continue
            else:
                other_vec_list = other_vector_json
                
            other_vec = np.array(other_vec_list, dtype='float32')
            other_norm = np.linalg.norm(other_vec)
            
            if curr_norm == 0 or other_norm == 0: continue
            
            # 코사인 유사도 계산
            similarity = np.dot(curr_vec, other_vec) / (curr_norm * other_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                target_cluster_id = cluster_id

        # 3. 할당 또는 생성
        reco = db.query(ArticleRecommend).filter(ArticleRecommend.id == reco_id).first()
        
        if target_cluster_id and best_similarity >= self.SIMILARITY_THRESHOLD:
            # 유사한 기사가 있으면 -> 그 기사의 그룹 ID를 그대로 사용 (합류)
            reco.article_cluster_id = target_cluster_id
            print(f"[클러스터링] ID {reco_id}: 기존 그룹 {target_cluster_id}번(유사도 {best_similarity:.4f})에 합류.")
        else:
            # 유사한 기사가 없으면 -> 자기 자신의 ID를 새로운 그룹 ID로 사용 (신규 생성)
            # (별도 테이블 없이도 유니크한 그룹 ID를 만드는 가장 쉬운 방법입니다)
            reco.article_cluster_id = reco_id
            print(f"[클러스터링] ID {reco_id}: 새로운 그룹 {reco_id}번 생성.")
        
        db.commit()

clustering_service = ClusteringService()