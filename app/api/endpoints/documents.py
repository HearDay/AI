from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from typing import List, Optional
from pydantic import BaseModel, Field

# 👇👇👇 이 부분을 수정합니다! (models.models -> app.models.document)
from app.core.database import get_db
from app.models.document import Article, ArticleRecommend, ArticleRecommendKeyword, ArticleRecommendVector
from app.services.keyword_extractor import keyword_extractor
from app.services.analysis_service import analysis_service

router = APIRouter(
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
    
    class Config:
        from_attributes = True

@router.post(
    "/process/article/{article_id}", 
    status_code=status.HTTP_202_ACCEPTED,
    summary="[백엔드용] 기사 ID를 받아 AI 분석 및 인덱싱"
)
async def process_document_by_id(
    article_id: int,
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

    reco.status = 'PROCESSING'
    await db.commit()

    try:
        keywords_list = keyword_extractor.extract(article.description, STANDARD_CANDIDATES)
        sbert_vector_np = analysis_service.encode_text(article.description)
        sbert_vector_list = sbert_vector_np.tolist() 

        await db.execute(
            ArticleRecommendKeyword.__table__.delete()\
            .where(ArticleRecommendKeyword.article_recommend_id == reco.id)
        )
        
        for kw in keywords_list:
            db.add(ArticleRecommendKeyword(article_recommend_id=reco.id, keyword=kw))
        
        await db.execute(
            ArticleRecommendVector.__table__.delete()\
            .where(ArticleRecommendVector.article_recommend_id == reco.id)
        )
        
        db.add(ArticleRecommendVector(
            article_recommend_id=reco.id, 
            sbert_vector=sbert_vector_list
        ))

        reco.status = 'COMPLETED'
        
        await db.commit()
        await db.refresh(reco)
        
        await analysis_service.add_document_to_index(
            reco_id=reco.id, 
            vector_list=sbert_vector_list
        )
        
        return {"message": "AI 분석 및 인덱싱 완료", "recommend_id": reco.id}

    except Exception as e:
        reco.status = 'FAILED'
        await db.commit()
        raise HTTPException(status_code=500, detail=f"AI 분석 중 오류 발생: {str(e)}")


@router.get(
    "/similar/article/{article_id}", 
    response_model=List[ArticleResponse], 
    summary="[SBERT 추천] 특정 기사와 유사한 기사 추천 (Faiss)"
)
async def get_similar_articles(
    article_id: int,
    db: AsyncSession = Depends(get_db)
):
    
    similar_article_ids = await analysis_service.find_similar_documents(db, article_id)
    
    if not similar_article_ids:
        return []
    
    query = select(Article).where(Article.id.in_(similar_article_ids))
    result = await db.execute(query)
    articles = result.scalars().all()
        
    return articles


@router.get(
    "/category/{category_name}", 
    response_model=List[ArticleResponse], 
    summary="[LLM 추천] 특정 카테고리 기사 목록 (콜드 스타트용)"
)
async def get_documents_by_category(
    category_name: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    
    query = (
        select(Article)
        .join(Article.recommend)
        .join(ArticleRecommend.keywords)
        .where(ArticleRecommendKeyword.keyword == category_name)
        .where(ArticleRecommend.status == 'COMPLETED')
        .order_by(Article.publish_date.desc())
        .limit(limit)
    )
    
    result = await db.execute(query)
    articles = result.scalars().unique().all()
    
    if not articles:
        raise HTTPException(
            status_code=404, 
            detail=f"'{category_name}' 카테고리의 기사를 찾을 수 없습니다."
        )
        
    return articles