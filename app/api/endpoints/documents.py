from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload, Session, selectinload
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.sql import text

from app.core.database import get_db, SessionLocalSync
from app.models.document import (
    Article, ArticleRecommend, ArticleRecommendKeyword, ArticleRecommendVector,
    UserCategory, UserRecentArticle
)
from app.services.keyword_extractor import keyword_extractor
from app.services.analysis_service import analysis_service
from app.services.bias_analyzer import bias_analyzer

router = APIRouter(tags=["AI Internal Processing"])
recommend_router = APIRouter(tags=["AI Recommendation"])

# 상수 정의
STANDARD_CANDIDATES = [
    "경제", "방송_연예", "IT", "쇼핑", "생활", "해외", "스포츠", "정치"
]
COLD_START_THRESHOLD = 10  # Cold Start 판단 기준 (읽은 기사 수)
DEFAULT_RECOMMENDATION_LIMIT = 5  # 기본 추천 개수

class ArticleResponse(BaseModel):
    id: int
    title: str
    origin_link: Optional[str] = None
    image_url: Optional[str] = None
    
    class Config:
        from_attributes = True

def to_article_response(article: Article) -> ArticleResponse:
    return ArticleResponse(
        id=article.id,
        title=article.title,
        origin_link=article.description,
        image_url=article.image_url
    )
    
# 추천 헬퍼 함수들
async def get_user_read_count(db: AsyncSession, user_id: int) -> int:
    """사용자가 읽은 기사 수 조회"""
    try:
        count_query = select(func.count(UserRecentArticle.id)).where(
            UserRecentArticle.user_id == user_id
        )
        result = await db.execute(count_query)
        return result.scalar_one_or_none() or 0
    except Exception as e:
        # 연결 문제 발생 시 재시도
        print(f"[Warning] get_user_read_count 오류 발생, 재시도: {e}")
        try:
            # 세션 새로고침 후 재시도
            await db.rollback()
            result = await db.execute(count_query)
            return result.scalar_one_or_none() or 0
        except Exception as retry_error:
            print(f"[Error] get_user_read_count 재시도 실패: {retry_error}")
            # 최종 실패 시 기본값 반환
            return 0


def build_base_article_query():
    """기본 기사 쿼리 빌더 (COMPLETED 상태, BIASED 제외)"""
    return (
        select(Article)
        .join(Article.recommend)
        .where(ArticleRecommend.status == 'COMPLETED')
        .where(ArticleRecommend.bias_label != 'BIASED')
    )


async def fill_with_random_articles(
    db: AsyncSession,
    existing_articles: List[Article],
    target_count: int = DEFAULT_RECOMMENDATION_LIMIT,
    category_name: Optional[str] = None
) -> List[Article]:
    """추천 뉴스가 target_count 미만일 때 랜덤 뉴스로 채워서 반환"""
    existing_ids = {article.id for article in existing_articles}
    needed_count = target_count - len(existing_articles)
    
    if needed_count <= 0:
        return existing_articles
    
    # 카테고리 지정 시 해당 카테고리 기사만 조회
    random_query = build_base_article_query()
    
    if category_name:
        random_query = (
            random_query
            .join(ArticleRecommend.keywords)
            .where(ArticleRecommendKeyword.keyword == category_name)
        )
    
    random_query = (
        random_query
        .where(~Article.id.in_(existing_ids) if existing_ids else True)
        .order_by(text("RAND()"))
        .limit(needed_count)
    )
    
    result = await db.execute(random_query)
    random_articles = result.scalars().all()
    
    return list(existing_articles) + list(random_articles)


async def get_random_category_articles(
    db: AsyncSession,
    category_name: str,
    limit: int = DEFAULT_RECOMMENDATION_LIMIT
) -> List[Article]:
    """특정 카테고리의 랜덤 기사 조회"""
    query = (
        build_base_article_query()
        .join(ArticleRecommend.keywords)
        .where(ArticleRecommendKeyword.keyword == category_name)
        .order_by(text("RAND()"))
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().unique().all())

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
        
        # 3. 벡터 생성 (편향 여부와 상관없이 항상 수행 - 클러스터링을 위해 필요)
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
            
        # 2. 벡터 저장 (편향 여부 상관없이 항상 저장)
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
                # 편향 기사 처리: FILTERED 상태로 변경
                reco_to_update.status = 'FILTERED'
                db_2.commit()
                print(f"[완료] ID {article_id}: FILTERED 저장 완료.")
            else:
                # 중립 기사 처리: COMPLETED 상태로 변경 및 Faiss 인덱싱
                reco_to_update.status = 'COMPLETED'
                db_2.commit()
                
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
    
    query = (
        build_base_article_query()
        .where(Article.id.in_(similar_article_ids))
    )
    result = await db.execute(query)
    articles = result.scalars().all()
    
    articles = await fill_with_random_articles(
        db, list(articles), target_count=DEFAULT_RECOMMENDATION_LIMIT
    )
    return [to_article_response(a) for a in articles]

@recommend_router.get(
    "/users/{user_id}/recommendations/category/{category_name}", 
    response_model=List[ArticleResponse],
    summary="[카테고리별 추천] 특정 카테고리의 사용자 맞춤 추천"
)
async def get_documents_by_categories(
    user_id: int,
    category_name: str,
    limit: int = DEFAULT_RECOMMENDATION_LIMIT,
    db: AsyncSession = Depends(get_db)
):
    read_count = await get_user_read_count(db, user_id)

    if read_count <= COLD_START_THRESHOLD:
        # Cold Start: 카테고리 기반 추천
        query = (
            build_base_article_query()
            .join(ArticleRecommend.keywords)
            .options(selectinload(Article.recommend).selectinload(ArticleRecommend.keywords))
            .where(ArticleRecommendKeyword.keyword == category_name)
            .order_by(Article.publish_date.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        articles = result.scalars().unique().all()

        # 기사가 없으면 랜덤으로 해당 카테고리 기사 조회
        if not articles:
            articles = await get_random_category_articles(db, category_name, limit)
            if not articles:
                # 그래도 없으면 일반 랜덤 기사로 채움
                articles = await fill_with_random_articles(db, [], target_count=limit)
            return [to_article_response(a) for a in articles]

        articles = await fill_with_random_articles(
            db, list(articles), target_count=limit, category_name=category_name
        )
        return [to_article_response(a) for a in articles]
    else:
        # Warm Start: SBERT 유사도 기반 + 카테고리 필터링
        similar_article_ids = await analysis_service.find_similar_documents_by_user(
            db, user_id, top_k=limit * 5
        )
        
        # 유사 기사가 없으면 랜덤으로 해당 카테고리 기사 조회
        if not similar_article_ids:
            articles = await get_random_category_articles(db, category_name, limit)
            if not articles:
                # 그래도 없으면 일반 랜덤 기사로 채움
                articles = await fill_with_random_articles(db, [], target_count=limit)
            return [to_article_response(a) for a in articles]

        query = (
            build_base_article_query()
            .join(ArticleRecommend.keywords)
            .options(selectinload(Article.recommend).selectinload(ArticleRecommend.keywords))
            .where(Article.id.in_(similar_article_ids))
            .where(ArticleRecommendKeyword.keyword == category_name)
        )
        result = await db.execute(query)
        articles = result.scalars().unique().all()

        # 카테고리 필터링 후 기사가 없으면 랜덤으로 해당 카테고리 기사 조회
        if not articles:
            articles = await get_random_category_articles(db, category_name, limit)
            if not articles:
                # 그래도 없으면 일반 랜덤 기사로 채움
                articles = await fill_with_random_articles(db, [], target_count=limit)
            return [to_article_response(a) for a in articles]

        # 유사도 순서 유지
        article_map = {article.id: article for article in articles}
        ordered_articles = [
            article_map[aid] for aid in similar_article_ids if aid in article_map
        ][:limit]

        ordered_articles = await fill_with_random_articles(
            db, ordered_articles, target_count=limit, category_name=category_name
        )
        return [to_article_response(a) for a in articles]

@recommend_router.get(
    "/users/{user_id}/recommendations", 
    response_model=List[ArticleResponse],
    summary="[메인 추천] 사용자 맞춤형 기사 추천 (LLM/SBERT 자동 전환)"
)
async def get_user_recommendations(
    user_id: int,
    limit: int = DEFAULT_RECOMMENDATION_LIMIT,
    db: AsyncSession = Depends(get_db)
):
    read_count = await get_user_read_count(db, user_id)

    if read_count <= COLD_START_THRESHOLD:
        # Cold Start: 사용자 선호 카테고리 기반 추천
        pref_query = select(UserCategory.user_category).where(
            UserCategory.user_id == user_id
        )
        user_categories = (await db.execute(pref_query)).scalars().all()

        if not user_categories:
            raise HTTPException(
                status_code=404,
                detail="사용자의 선호 카테고리 정보를 찾을 수 없습니다."
            )

        query = (
            build_base_article_query()
            .join(ArticleRecommend.keywords)
            .where(ArticleRecommendKeyword.keyword.in_(user_categories))
            .order_by(Article.publish_date.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        articles = result.scalars().unique().all()

        articles = await fill_with_random_articles(db, list(articles), target_count=limit)
        return [to_article_response(a) for a in articles]
    else:
        # Warm Start: SBERT 유사도 기반 추천
        similar_article_ids = await analysis_service.find_similar_documents_by_user(
            db, user_id, top_k=limit
        )
        if not similar_article_ids:
            return []

        query = (
            build_base_article_query()
            .where(Article.id.in_(similar_article_ids))
        )
        result = await db.execute(query)
        articles = result.scalars().all()

        # 유사도 순서 유지
        article_map = {article.id: article for article in articles}
        ordered_articles = [
            article_map[aid] for aid in similar_article_ids if aid in article_map
        ]

        ordered_articles = await fill_with_random_articles(
            db, ordered_articles, target_count=limit
        )
        return [to_article_response(a) for a in articles]