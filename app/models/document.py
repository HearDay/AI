from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, UniqueConstraint
from sqlalchemy.sql import func
from app.core.database import Base

class Document(Base):
    __tablename__ = "documents"

    # 1. 우리 시스템의 기본 ID
    id = Column(Integer, primary_key=True, index=True)

    # 2. 백엔드에서 받는 기사 정보 필드들 (추가됨)
    article_id = Column(String, unique=True, index=True, nullable=False)
    original_url = Column(String, unique=True, nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=False)
    title = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
    category = Column(String, index=True) # 원본 카테고리
    
    # 3. AI 분석 결과 필드들
    keywords = Column(JSON)      # LLM 키워드
    sbert_vector = Column(JSON)  # SBERT 벡터
    
    # 4. 생성 시간
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 5. 중복 방지를 위한 제약 조건 추가 (선택사항이지만 권장)
    __table_args__ = (
        UniqueConstraint('article_id', name='uq_article_id'),
        UniqueConstraint('original_url', name='uq_original_url'),
    )