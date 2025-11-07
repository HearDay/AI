from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Index, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum

# --- 1. 백엔드가 관리하는 'Article' 테이블 ---
# (우리는 이 테이블을 읽기만 합니다)
class Article(Base):
    __tablename__ = "article"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    article_category = Column(String(100))
    description = Column(Text, nullable=False) # (기사 본문)
    image_url = Column(String(2083))
    origin_link = Column(String(2083), nullable=False)
    publish_date = Column(DateTime(timezone=True), nullable=False)
    title = Column(Text, nullable=False)
    
    # 1:1 관계 (Article <-> ArticleDetail) - 백엔드 스키마에 따름
    article_detail_id = Column(Integer, ForeignKey("article_detail.id")) # (가정: article_detail 테이블이 존재)
    
    # 1:1 관계 (Article <-> ArticleRecommend)
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="article", uselist=False)

# --- 2. AI가 관리하는 'ArticleRecommend' (작업 상태) ---
class ArticleRecommend(Base):
    __tablename__ = "article_recommend"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Enum/String 타입으로 상태 관리
    status = Column(String(50), default='PENDING', index=True, nullable=False) 
    
    # 1:1 관계 (Article 테이블에서 이 레코드를 참조)
    article = relationship("Article", back_populates="recommend", uselist=False)
    
    # 1:N 관계 (이 레코드는 여러 개의 키워드를 가짐)
    keywords = relationship("ArticleRecommendKeyword", back_populates="recommend")
    
    # 1:1 관계 (이 레코드는 하나의 벡터를 가짐)
    vector = relationship("ArticleRecommendVector", back_populates="recommend", uselist=False)

# --- 3. AI가 관리하는 'ArticleRecommendKeyword' (LLM 키워드) ---
class ArticleRecommendKeyword(Base):
    __tablename__ = "article_recommend_keywords"
    
    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(100), index=True)
    
    # N:1 관계 (키워드는 하나의 Recommend 레코드에 속함)
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="keywords")

# --- 4. AI가 관리하는 'ArticleRecommendVector' (SBERT 벡터) ---
class ArticleRecommendVector(Base):
    __tablename__ = "article_recommend_vector"
    
    id = Column(Integer, primary_key=True, index=True)
    sbert_vector = Column(JSON, nullable=False) # 벡터는 JSON 타입으로 저장
    
    # 1:1 관계 (벡터는 하나의 Recommend 레코드에 속함)
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="vector")

# (참고: 백엔드 스키마에 'article_detail' 테이블이 언급되어 있으나, AI 분석에는 필요하지 않아 정의하지 않았습니다.)
# (또한, AWS RDS 테이블 자동 생성을 위해 main.py가 이 파일을 임포트해야 합니다)