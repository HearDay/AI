from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Index, UniqueConstraint, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base 

# --- 1. 백엔드가 관리하는 'Article' 테이블 ---
class Article(Base):
    __tablename__ = "article"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    article_category = Column(String(100))
    description = Column(Text, nullable=False)
    image_url = Column(String(2083))
    origin_link = Column(String(2083), nullable=False)
    publish_date = Column(DateTime(timezone=True), nullable=False)
    title = Column(Text, nullable=False)
    
    article_detail_id = Column(Integer) 
    
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="article", uselist=False)

    __table_args__ = (
        Index('uq_original_url_prefix', 'origin_link', unique=True, mysql_length=255),
    )

# --- 2. AI가 관리하는 'ArticleRecommend' ---
class ArticleRecommend(Base):
    __tablename__ = "article_recommend"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 상태: PENDING, PROCESSING, COMPLETED, FAILED, FILTERED
    status = Column(String(50), default='PENDING', index=True, nullable=False) 
    
    # 편향성 분석 결과
    bias_label = Column(String(50), default='UNKNOWN', index=True)
    bias_score = Column(Float, default=0.0)
    article_cluster_id= Column(Integer, index=True)
    # [제거됨] article_cluster_id 및 cluster 관계 삭제

    article = relationship("Article", back_populates="recommend", uselist=False)
    keywords = relationship("ArticleRecommendKeyword", back_populates="recommend")
    vector = relationship("ArticleRecommendVector", back_populates="recommend", uselist=False)

# --- 3. Keywords 테이블 ---
class ArticleRecommendKeyword(Base):
    __tablename__ = "article_recommend_keywords"
    
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"), primary_key=True)
    keyword = Column(String(100), primary_key=True, index=True)
    
    recommend = relationship("ArticleRecommend", back_populates="keywords")

# --- 4. Vector 테이블 ---
class ArticleRecommendVector(Base):
    __tablename__ = "article_recommend_vector"
    
    id = Column(Integer, primary_key=True, index=True)
    sbert_vector = Column(JSON, nullable=False)
    
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="vector")

    __table_args__ = (
        UniqueConstraint('article_recommend_id', name='uq_reco_id_vector'),
    )

# --- 5. User 관련 테이블 ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    categories = relationship("UserCategory", back_populates="user")
    recent_articles = relationship("UserRecentArticle", back_populates="user")

class UserCategory(Base):
    __tablename__ = "user_category"
    id = Column(Integer, primary_key=True, index=True)
    user_category = Column(String(100))
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="categories")

class UserRecentArticle(Base):
    __tablename__ = "user_recent_article"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="recent_articles")
    article_id = Column(Integer, ForeignKey("article.id"))