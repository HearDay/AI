from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload, Session
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy import func
import asyncio

# DB 및 모델 임포트
from app.core.database import get_db, SessionLocal, SessionLocalSync
from app.models.document import (
    Article, ArticleRecommend, ArticleRecommendKeyword, ArticleRecommendVector,
    User, UserCategory, UserRecentArticle
)
# 서비스 임포트
from app.services.keyword_extractor import keyword_extractor
from app.services.analysis_service import analysis_service
from app.services.bias_analyzer import bias_analyzer
from app.services.clustering_service import clustering_service # [복구] 클러스터링 서비스 임포트

router = APIRouter(tags=["AI Internal Processing"])
recommend_router = APIRouter(tags=["AI Recommendation"])

STANDARD_CANDIDATES = [
    "경제", "방송_연예", "IT", "쇼핑", "생활", "해외", "스포츠", "정치"
]

class ArticleResponse(BaseModel):
    id: int
    title: str
    origin_link: str
    image_url: Optional[str] = None
    
    class Config:
        from_attributes = True

# --- 백그라운드 작업 헬퍼 함수 (동기) ---

def set_status_failed_sync(reco_id: int):
    db_fail: Session = SessionLocalSync()
    try:
        reco_fail = db_fail.query(ArticleRecommend).filter(ArticleRecommend.id == reco_id).first()
        if reco_fail:
            reco_fail.status = 'FAILED'
            db_fail.commit()
    except Exception as e:
        print(f"[set_status_failed 오류] ID {reco_id} 상태 변경 실패: {repr(e)}")
    finally:
        db_fail.close()

# --- 백그라운드 AI 작업 함수 (완전 동기) ---
def process_ai_task_background(article_id: int):
    print(f"[백그라운드 작업 시작] Article ID: {article_id}")
    
    article_text: Optional[str] = None
    article_title: Optional[str] = None
    reco_id: Optional[int] = None
    
    # [1단계] DB 조회 및 상태 변경 (동기 세션)
    db: Session = SessionLocalSync()
    try:
        article = db.query(Article).options(joinedload(Article.recommend))\
                    .filter(Article.id == article_id).first()
        
        if not (article and article.recommend):
            print(f"[백그라운드 오류] ID {article_id}: 기사 또는 추천 정보를 찾을 수 없습니다.")
            return
            
        reco = article.recommend
        
        # 이미 처리된 작업이면 스킵
        if reco.status in ['COMPLETED', 'PROCESSING', 'FILTERED']:
             print(f"[백그라운드] ID {article_id}: 이미 처리된 작업입니다. (Status: {reco.status})")
             db.close()
             return

        reco.status = 'PROCESSING'
        db.commit()
        
        article_text = article.description
        article_title = article.title
        reco_id = reco.id
        
    except Exception as e_fetch:
        print(f"[백그라운드 실패] ID {article_id} (1단계 DB 조회 중) 오류 발생: {repr(e_fetch)}")
        db.rollback()
        return 
    finally:
        db.close()

    # [2단계] AI 분석 (키워드, 편향성, 벡터)
    keywords_list = []
    sbert_vector_list = []
    bias_result = {"label": "UNKNOWN", "score": 0.0}
    is_biased = False

    try:
        if not article_text or not reco_id:
            raise ValueError("1단계에서 기사 정보(text, reco_id)를 가져오지 못했습니다.")
            
        # 1. 키워드 추출
        keywords_list = keyword_extractor.extract(article_text, STANDARD_CANDIDATES)
        
        # 2. 편향성 분석 (모든 기사 대상)
        try:
            # 동기 함수 호출 (await 제거)
            bias_result = bias_analyzer.analyze_bias(article_text)
        except Exception as e:
            print(f"[Warning] 편향성 분석 오류: {e}")
            bias_result = {"label": "UNKNOWN", "score": 0.0}

        print(f"[AI 분석] ID {article_id} 편향성 결과: {bias_result['label']} (Score: {bias_result['score']:.2f})")

        if bias_result['label'] == "BIASED":
            is_biased = True
        
        # 3. [수정됨] 벡터 생성 (편향 여부와 상관없이 항상 수행!)
        # 클러스터링을 하려면 편향된 기사도 벡터가 필요합니다.
        sbert_vector_np = analysis_service.encode_text(article_text)
        sbert_vector_list = sbert_vector_np.tolist()
            
    except Exception as e_ai:
        print(f"[백그라운드 실패] ID {article_id} (2단계 AI 분석 중) 오류 발생: {repr(e_ai)}")
        set_status_failed_sync(reco_id) 
        return 
        
    # [3단계] 결과 저장 및 인덱싱/클러스터링 (동기 세션)
    db_2: Session = SessionLocalSync() 
    try:
        # 1. 키워드 저장
        db_2.query(ArticleRecommendKeyword)\
            .filter(ArticleRecommendKeyword.article_recommend_id == reco_id)\
            .delete()
            
        for kw in keywords_list:
            db_2.add(ArticleRecommendKeyword(article_recommend_id=reco_id, keyword=kw))
            
        # 2. [수정됨] 벡터 저장 (편향 여부 상관없이 항상 저장)
        db_2.query(ArticleRecommendVector)\
            .filter(ArticleRecommendVector.article_recommend_id == reco_id)\
            .delete()
        
        db_2.add(ArticleRecommendVector(
            article_recommend_id=reco_id, 
            sbert_vector=sbert_vector_list
        ))

        # 3. 상태 및 편향성 정보 업데이트
        reco_to_update = db_2.query(ArticleRecommend).filter(ArticleRecommend.id == reco_id).first()
        
        if reco_to_update:
            reco_to_update.bias_label = bias_result['label']
            reco_to_update.bias_score = bias_result['score']
            
            # 변경 사항 1차 저장 (커밋)
            db_2.commit()
            
            if is_biased:
                # [편향 기사 처리]
                # 1. 상태를 FILTERED로 변경
                reco_to_update.status = 'FILTERED'
                db_2.commit()
                
                # 2. 클러스터링 서비스 호출 (여기서 article_cluster_id가 생성됨)
                print(f"[클러스터링] ID {article_id}: 편향 기사 그룹화 시작...")
                clustering_service.assign_to_cluster(
                    db_2, 
                    reco_id, 
                    sbert_vector_list, 
                    article_title
                )
                print(f"[완료] ID {article_id}: FILTERED 저장 및 클러스터링 완료.")
                
            else:
                # [중립 기사 처리]
                reco_to_update.status = 'COMPLETED'
                db_2.commit()
                
                # Faiss 인덱싱 (일반 추천용)
                analysis_service.add_document_to_index(reco_id, sbert_vector_list)
                print(f"[완료] ID {article_id}: COMPLETED 및 인덱싱 완료.")

    except Exception as e_update:
        print(f"[백그라운드 실패] ID {article_id} (3단계 DB 저장 중) 오류 발생: {repr(e_update)}")
        db_2.rollback()
        set_status_failed_sync(reco_id)
    finally:
        db_2.close()


# --- API Endpoints (비동기) ---

@router.post(
    "/process/article/{article_id}", 
    status_code=status.HTTP_202_ACCEPTED,
    summary="[백엔드용] 기사 ID를 받아 AI 분석 작업을 '예약'"
)
async def process_document_by_id(
    article_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)  
):
    query = select(Article).options(joinedload(Article.recommend))\
            .where(Article.id == article_id)
    result = await db.execute(query)
    article = result.scalars().first()

    if not article:
        raise HTTPException(status_code=404, detail="Article을 찾을 수 없습니다.")
    if not article.recommend:
        raise HTTPException(status_code=404, detail="ArticleRecommend 레코드가 연결되지 않았습니다.")

    reco = article.recommend
    
    if reco.status in ['COMPLETED', 'PROCESSING', 'FILTERED']:
        return {"message": f"Already processed (Status: {reco.status})"}

    background_tasks.add_task(process_ai_task_background, article_id)
    
    return {"message": "AI 분석 작업이 백그라운드에서 시작되었습니다."}


@recommend_router.get(
    "/similar/article/{article_id}", 
    response_model=List[ArticleResponse], 
    summary="[SBERT] 특정 기사와 유사한 기사 추천"
)
async def get_similar_articles(
    article_id: int,
    db: AsyncSession = Depends(get_db)
):
    similar_article_ids = await analysis_service.find_similar_documents(db, article_id)
    
    if not similar_article_ids:
        return []
    
    # 필터 추가: 편향된(FILTERED, BIASED) 기사는 추천에서 제외
    query = select(Article).join(Article.recommend)\
            .where(Article.id.in_(similar_article_ids))\
            .where(ArticleRecommend.bias_label != 'BIASED')

    result = await db.execute(query)
    articles = result.scalars().all()
        
    return articles

@recommend_router.get(
    "/category/", 
    response_model=List[ArticleResponse],
    summary="[LLM] 여러 카테고리 기사 목록 (다중 입력)"
)
async def get_documents_by_categories(
    categories: List[str] = Query(..., description="검색할 카테고리 목록"),
    limit: int = 20, 
    db: AsyncSession = Depends(get_db)
):
    query = select(Article).join(Article.recommend).join(ArticleRecommend.keywords)\
            .where(ArticleRecommendKeyword.keyword.in_(categories))\
            .where(ArticleRecommend.status == 'COMPLETED')\
            .where(ArticleRecommend.bias_label != 'BIASED')\
            .order_by(Article.publish_date.desc()).limit(limit)
    result = await db.execute(query)
    articles = result.scalars().unique().all()
    
    if not articles:
        raise HTTPException(status_code=404, detail="선택한 카테고리에 해당하는 기사를 찾을 수 없습니다.")
    return articles

@recommend_router.get(
    "/users/{user_id}/recommendations", 
    response_model=List[ArticleResponse],
    summary="[메인 추천] 사용자 맞춤형 기사 추천 (LLM/SBERT 자동 전환)"
)
async def get_user_recommendations(
    user_id: int, 
    limit: int = 5, 
    db: AsyncSession = Depends(get_db)
):
    # 1. 사용자가 읽은 기사 수 확인
    count_query = select(func.count(UserRecentArticle.id)).where(UserRecentArticle.user_id == user_id)
    read_count = (await db.execute(count_query)).scalar_one_or_none() or 0

    if read_count <= 10:
        # 2. (LLM 로직) Cold Start
        pref_query = select(UserCategory.user_category).where(UserCategory.user_id == user_id)
        user_categories = (await db.execute(pref_query)).scalars().all()
        
        if not user_categories:
            raise HTTPException(status_code=404, detail="사용자의 선호 카테고리 정보를 찾을 수 없습니다.")
        
        llm_query = select(Article).join(Article.recommend).join(ArticleRecommend.keywords)\
                    .where(ArticleRecommendKeyword.keyword.in_(user_categories))\
                    .where(ArticleRecommend.status == 'COMPLETED')\
                    .where(ArticleRecommend.bias_label != 'BIASED')\
                    .order_by(Article.publish_date.desc()).limit(limit)
        result = await db.execute(llm_query)
        return result.scalars().unique().all()
    else:
        # 3. (SBERT 로직) Warm Start
        similar_article_ids = await analysis_service.find_similar_documents_by_user(db, user_id, top_k=limit)
        if not similar_article_ids: return []
        
        # SBERT 추천 결과에서도 편향된 기사는 제외
        sbert_query = select(Article).join(Article.recommend)\
                      .where(Article.id.in_(similar_article_ids))\
                      .where(ArticleRecommend.bias_label != 'BIASED')

        result = await db.execute(sbert_query)
        articles = result.scalars().all()
        
        article_map = {article.id: article for article in articles}
        ordered_articles = [
            article_map[aid] for aid in similar_article_ids if aid in article_map
        ]
        return ordered_articles

@recommend_router.get(
    "/politics/neutral", 
    response_model=List[ArticleResponse],
    summary="[정치] 중립적인 뉴스 추천"
)
async def get_neutral_political_news(limit: int = 10, db: AsyncSession = Depends(get_db)):
    query = (
        select(Article)
        .join(Article.recommend)
        .join(ArticleRecommend.keywords)
        .where(ArticleRecommendKeyword.keyword == "정치")     
        .where(ArticleRecommend.bias_label == "NEUTRAL")     
        .where(ArticleRecommend.status == 'COMPLETED')      
        .order_by(Article.publish_date.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    articles = result.scalars().unique().all()
    
    if not articles:
        raise HTTPException(status_code=404, detail="중립적인 정치 뉴스가 없습니다.")
        
    return articles