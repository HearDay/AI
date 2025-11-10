from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from typing import List
from pydantic import BaseModel
from sqlalchemy import func

# 백그라운드 작업에서 새 DB 세션을 만들기 위해 SessionLocal을 임포트합니다.
from app.core.database import get_db, SessionLocal
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

# --- 백그라운드에서 실제 AI 작업을 수행할 별도의 함수 ---
async def process_ai_task_background(article_id: int):
    """
    백그라운드에서 실행될 실제 AI 분석 및 DB 업데이트 작업
    """
    print(f"[백그라운드 작업 시작] Article ID: {article_id}")
    
    # ✅ 수정: async with로 세션 생성
    async with SessionLocal() as db:
        reco = None  # except 블록에서 사용하기 위해 미리 선언
        try:
            # 2. DB에서 기사 및 추천 정보 조회
            query = select(Article).options(joinedload(Article.recommend))\
                    .where(Article.id == article_id)
            result = await db.execute(query)
            article = result.scalars().first()
            
            if not (article and article.recommend):
                print(f"[백그라운드 오류] ID {article_id}의 기사 또는 추천 정보를 찾을 수 없습니다.")
                return # 작업 종료

            reco = article.recommend
            
            # 3. 'PROCESSING'으로 상태 변경
            reco.status = 'PROCESSING'
            await db.commit()
            print(f"[백그라운드] ID {article_id} 처리 중...")

            # 4. AI 분석 (서비스 함수들은 async def여야 함)
            keywords_list = await keyword_extractor.extract(article.description, STANDARD_CANDIDATES)
            sbert_vector_np = await analysis_service.encode_text(article.description)
            
            # ✅ numpy array를 Python list로 변환 (JSON 직렬화 가능)
            sbert_vector_list = sbert_vector_np.tolist()
            
            # ✅ 디버깅: 벡터 정보 출력
            print(f"[벡터 정보] 타입: {type(sbert_vector_list)}, 길이: {len(sbert_vector_list)}")
            print(f"[벡터 샘플] 첫 3개 값: {sbert_vector_list[:3]}")
            
            # ✅ 타입 검증: list인지 확인
            if not isinstance(sbert_vector_list, list):
                raise ValueError(f"벡터가 list 타입이 아닙니다: {type(sbert_vector_list)}")

            # --- 키워드 저장 ---
            await db.execute(
                ArticleRecommendKeyword.__table__.delete()\
                .where(ArticleRecommendKeyword.article_recommend_id == reco.id)
            )
            for kw in keywords_list:
                db.add(ArticleRecommendKeyword(article_recommend_id=reco.id, keyword=kw))
            
            # --- 벡터 저장 ---
            # ✅ 기존 벡터 삭제
            await db.execute(
                ArticleRecommendVector.__table__.delete()\
                .where(ArticleRecommendVector.article_recommend_id == reco.id)
            )
            await db.flush()  # 삭제 즉시 반영
            
            # ✅ 새 벡터 추가 (list를 JSON으로 자동 변환됨)
            new_vector = ArticleRecommendVector(
                article_recommend_id=reco.id, 
                sbert_vector=sbert_vector_list  # list[float]를 그대로 저장
            )
            db.add(new_vector)
            await db.flush()  # 저장 즉시 반영
            
            print(f"[벡터 저장 완료] ArticleRecommend ID: {reco.id}")

            # --- 상태 완료 ---
            reco.status = 'COMPLETED'
            await db.commit()
            await db.refresh(reco)
            
            # --- Faiss 인덱스 추가 ---
            await analysis_service.add_document_to_index(
                reco_id=reco.id, 
                vector_list=sbert_vector_list
            )
            print(f"[백그라운드 성공] ID {article_id} AI 분석 완료.")

        except Exception as e:
            # 5. 롤백 및 에러 로깅
            print(f"[백그라운드 실패] ID {article_id} 분석 중 오류 발생: {repr(e)}")
            import traceback
            print(f"[상세 에러]\n{traceback.format_exc()}")  # 전체 스택 트레이스 출력
            
            await db.rollback()  # ✅ 추가: 명시적 롤백
            
            # reco가 존재하면 FAILED 상태로 업데이트
            try:
                if reco is not None:
                    reco.status = 'FAILED'
                    await db.commit()
                    print(f"[상태 업데이트] ID {article_id} -> FAILED")
            except Exception as update_error:
                print(f"[백그라운드] 상태 업데이트 실패: {repr(update_error)}")

# --- API 엔드포인트 정의 ---

@router.post(
    "/process/article/{article_id}", 
    status_code=status.HTTP_202_ACCEPTED,
    summary="[백그라운드] 기사 ID를 받아 AI 분석 작업을 '예약'"
)
async def process_document_by_id(
    article_id: int,
    background_tasks: BackgroundTasks, # 1. BackgroundTasks 주입
    db: AsyncSession = Depends(get_db)  # 2. DB 세션은 '상태 확인'용으로만 사용
):
    """
    [수정됨] 이 API는 AI 분석을 '즉시 실행'하지 않고,
    '백그라운드 작업'으로 예약한 뒤 사용자에게 202 응답을 즉시 반환합니다.
    """
    
    # 3. API는 DB에서 '상태 확인'만 빠르게 수행합니다.
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

    # 4. 무거운 AI 작업을 백그라운드로 넘깁니다.
    background_tasks.add_task(process_ai_task_background, article_id)
    
    # 5. AI 작업이 끝나길 기다리지 않고, 즉시 '접수 완료' 응답을 보냅니다.
    return {"message": "AI 분석 작업이 백그라운드에서 시작되었습니다."}


# --- (GET API들은 변경 없음) ---

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
    "/category", 
    response_model=List[ArticleResponse], 
    summary="[LLM 추천] 여러 카테고리 기반 기사 목록 (콜드 스타트용)"
)
async def get_documents_by_categories(
    names: str,  # 예: "경제,IT,스포츠"
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    여러 카테고리를 한 번에 받아서, 
    일치하는 카테고리 키워드 개수가 많은 기사 순으로 반환합니다.
    예: /category?names=경제,IT,스포츠
    """

    # 1️⃣ 입력된 카테고리 문자열을 리스트로 분리
    category_list = [name.strip() for name in names.split(",") if name.strip()]
    if not category_list:
        raise HTTPException(status_code=400, detail="카테고리 이름을 하나 이상 입력해야 합니다.")

    # 2️⃣ ArticleRecommend, ArticleRecommendKeyword 조인 후, 일치 개수 계산
    query = (
        select(
            Article,
            func.count(ArticleRecommendKeyword.keyword).label("match_count")  # 일치 개수 계산
        )
        .join(Article.recommend)
        .join(ArticleRecommend.keywords)
        .where(ArticleRecommendKeyword.keyword.in_(category_list))
        .where(ArticleRecommend.status == 'COMPLETED')
        .group_by(Article.id)
        .order_by(func.count(ArticleRecommendKeyword.keyword).desc(), Article.publish_date.desc())
        .limit(limit)
    )

    # 3️⃣ 실행
    result = await db.execute(query)
    articles = [row[0] for row in result.fetchall()]  # row[0]은 Article 객체

    if not articles:
        raise HTTPException(
            status_code=404, 
            detail=f"{category_list} 중 일치하는 기사를 찾을 수 없습니다."
        )

    return articles