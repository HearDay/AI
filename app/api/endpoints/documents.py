from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from typing import List, Optional
from pydantic import BaseModel, Field
import datetime
from sqlalchemy import func, desc 

from app.core.database import get_db, SessionLocal
from app.models.document import (
    Article, ArticleRecommend, ArticleRecommendKeyword, ArticleRecommendVector,
    User, UserCategory, UserRecentArticle
)
from app.services.keyword_extractor import keyword_extractor
from app.services.analysis_service import analysis_service

router = APIRouter(
    tags=["AI Internal Processing"]
)
recommend_router = APIRouter(
    tags=["AI Recommendation"]
)

STANDARD_CANDIDATES = [
    "경제",
    "방송 / 연예",
    "IT",
    "쇼핑",
    "생활",
    "해외",
    "스포츠",
    "정치"
]

class ArticleResponse(BaseModel):
    id: int
    title: str
    origin_link: str
    image_url: str
    
    class Config:
        from_attributes = True

async def set_status_failed(reco_id: int):
    db_fail: AsyncSession = SessionLocal()
    try:
        reco_fail = await db_fail.get(ArticleRecommend, reco_id)
        if reco_fail:
            reco_fail.status = 'FAILED'
            await db_fail.commit()
    except Exception as e:
        print(f"[set_status_failed 오류] ID {reco_id} 상태 변경 실패: {repr(e)}")
    finally:
        await db_fail.close()

async def process_ai_task_background(article_id: int):
    print(f"[백그라운드 작업 시작] Article ID: {article_id}")
    article_text: Optional[str] = None
    reco_id: Optional[int] = None
    db_session_1: AsyncSession = SessionLocal()
    try:
        query = select(Article).options(joinedload(Article.recommend)).where(Article.id == article_id)
        result = await db_session_1.execute(query)
        article = result.scalars().first()
        if not (article and article.recommend):
            print(f"[백그라운드 오류] ID {article_id}의 기사 또는 추천 정보를 찾을 수 없습니다.")
            return
        reco = article.recommend
        if reco.status == 'COMPLETED' or reco.status == 'PROCESSING':
             print(f"[백그라운드] ID {article_id}: 이미 처리되었거나 처리 중인 작업입니다. (Status: {reco.status})")
             await db_session_1.close()
             return
        reco.status = 'PROCESSING'
        await db_session_1.commit()
        article_text = article.description
        reco_id = reco.id
    except Exception as e_fetch:
        print(f"[백그라운드 실패] ID {article_id} (1단계 DB 조회 중) 오류 발생: {repr(e_fetch)}")
        await db_session_1.rollback()
        return 
    finally:
        if db_session_1.is_active:
             await db_session_1.close()
    try:
        if not article_text or not reco_id:
            raise ValueError("1단계에서 기사 정보(text, reco_id)를 가져오지 못했습니다.")
        keywords_list = await keyword_extractor.extract(article_text, STANDARD_CANDIDATES)
        sbert_vector_np = await analysis_service.encode_text(article_text)
        sbert_vector_list = sbert_vector_np.tolist()
    except Exception as e_ai:
        print(f"[백그라운드 실패] ID {article_id} (2단계 AI 분석 중) 오류 발생: {repr(e_ai)}")
        await set_status_failed(reco_id) 
        return 
    db_session_2: AsyncSession = SessionLocal() 
    try:
        await db_session_2.execute(
            ArticleRecommendKeyword.__table__.delete().where(ArticleRecommendKeyword.article_recommend_id == reco_id)
        )
        for kw in keywords_list:
            db_session_2.add(ArticleRecommendKeyword(article_recommend_id=reco_id, keyword=kw))
        await db_session_2.execute(
            ArticleRecommendVector.__table__.delete().where(ArticleRecommendVector.article_recommend_id == reco_id)
        )
        db_session_2.add(ArticleRecommendVector(
            article_recommend_id=reco_id, 
            sbert_vector=sbert_vector_list
        ))
        reco_to_update = await db_session_2.get(ArticleRecommend, reco_id)
        if reco_to_update:
            reco_to_update.status = 'COMPLETED'
        await db_session_2.commit()
        await analysis_service.add_document_to_index(
            reco_id=reco_id, 
            vector_list=sbert_vector_list
        )
    except Exception as e_update:
        print(f"[백그라운드 실패] ID {article_id} (3단계 DB 업데이트 중) 오류 발생: {repr(e_update)}")
        await db_session_2.rollback()
        await set_status_failed(reco_id)
    finally:
        await db_session_2.close()

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
    
    if reco.status == 'COMPLETED':
        return {"message": "이미 처리된 기사입니다."}
    if reco.status == 'PROCESSING':
        return {"message": "현재 처리 중인 기사입니다."}

    background_tasks.add_task(process_ai_task_background, article_id)
    
    return {"message": "AI 분석 작업이 백그라운드에서 시작되었습니다."}

@recommend_router.get(
    "/similar/article/{article_id}", 
    response_model=List[ArticleResponse], 
    summary="[SBERT 추천] 특정 기사와 유사한 기사 추천 (Faiss)"
)
async def get_similar_articles(
    article_id: int,
    db: AsyncSession = Depends(get_db)
):
    similar_article_ids = await analysis_service.find_similar_documents_by_article(db, article_id)
    
    if not similar_article_ids:
        return []
    
    query = select(Article).where(Article.id.in_(similar_article_ids))
    result = await db.execute(query)
    articles = result.scalars().all()
        
    return articles

@recommend_router.get(
    "/category/", 
    response_model=List[ArticleResponse], 
    summary="[LLM 추천] 여러 카테고리 기사 목록 (다중 입력)"
)
async def get_documents_by_categories(
    categories: List[str] = Query(..., description="검색할 카테고리 목록"),
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    query = (
        select(Article)
        .join(Article.recommend)
        .join(ArticleRecommend.keywords)
        .where(ArticleRecommendKeyword.keyword.in_(categories))
        .where(ArticleRecommend.status == 'COMPLETED')
        .order_by(Article.publish_date.desc())
        .limit(limit)
    )
    
    result = await db.execute(query)
    articles = result.scalars().unique().all()
    
    if not articles:
        raise HTTPException(
            status_code=404, 
            detail="선택한 카테고리에 해당하는 기사를 찾을 수 없습니다."
        )
        
    return articles

@recommend_router.get(
    "/users/{user_id}/recommendations",
    response_model=List[ArticleResponse],
    summary="[최종 추천] 사용자 맞춤형 기사 추천 (LLM/SBERT 자동 전환)"
)
async def get_user_recommendations(
    user_id: int,
    limit: int = 5,
    db: AsyncSession = Depends(get_db)
):
    """
    사용자 ID를 받아, 읽은 기사 수에 따라 
    LLM(카테고리) 또는 SBERT(유사도) 기반 추천을 자동으로 반환합니다.
    
    - 10개 이하: 선호 카테고리 기반 (LLM)
    - 10개 초과: 읽은 기사들의 평균 벡터 기반 (SBERT)
    """
    
    # 1. 사용자가 읽은 기사 수 확인
    count_query = select(func.count(UserRecentArticle.id))\
                    .where(UserRecentArticle.user_id == user_id)
    read_count = (await db.execute(count_query)).scalar_one_or_none() or 0

    if read_count <= 10:
        # 2. (LLM 로직) 10개 이하: 선호 카테고리 기반 추천
        print(f"User {user_id}: LLM 기반 추천 (읽은 기사 {read_count}개)")
        
        pref_query = select(UserCategory.category_name)\
                     .where(UserCategory.user_id == user_id)
        user_categories = (await db.execute(pref_query)).scalars().all()

        if not user_categories:
            raise HTTPException(status_code=404, detail="사용자의 선호 카테고리 정보를 찾을 수 없습니다.")
        
        llm_query = (
            select(Article)
            .join(Article.recommend)
            .join(ArticleRecommend.keywords)
            .where(ArticleRecommendKeyword.keyword.in_(user_categories))
            .where(ArticleRecommend.status == 'COMPLETED')
            .order_by(Article.publish_date.desc())
            .limit(limit)
        )
        result = await db.execute(llm_query)
        articles = result.scalars().unique().all()
        
        if not articles:
            raise HTTPException(status_code=404, detail="선호 카테고리에 맞는 기사가 없습니다.")
        
        return articles

    else:
        # 3. (SBERT 로직) 10개 초과: 읽은 모든 기사의 평균 벡터 기반 추천
        print(f"User {user_id}: SBERT 기반 추천 (읽은 기사 {read_count}개)")

        # ✨ 수정: find_similar_documents_by_user 사용
        # 이 메서드는 사용자가 읽은 모든 기사의 평균 벡터를 계산하여 추천
        similar_article_ids = await analysis_service.find_similar_documents_by_user(
            db, user_id, top_k=limit
        )
    
        if not similar_article_ids:
            return []
        
        sbert_query = select(Article).where(Article.id.in_(similar_article_ids))
        result = await db.execute(sbert_query)
        articles = result.scalars().all()

        article_map = {article.id: article for article in articles}
        ordered_articles = [
            article_map[article_id] 
            for article_id in similar_article_ids  # AnalysisService가 반환한 순서 유지
            if article_id in article_map
        ]

        return ordered_articles