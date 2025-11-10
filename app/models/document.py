from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, UniqueConstraint, LargeBinary, TypeDecorator, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import numpy as np
import json
from datetime import datetime

# ✅ Numpy 배열을 BLOB으로 저장하는 커스텀 타입
class NumpyArray(TypeDecorator):
    """
    Numpy 배열을 바이너리(BLOB)로 저장
    - 저장: numpy array -> bytes
    - 조회: bytes -> numpy array
    """
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Python numpy array -> bytes (DB 저장 시)"""
        if value is None:
            return None
        # numpy array를 bytes로 변환
        if isinstance(value, np.ndarray):
            return value.tobytes()
        # list인 경우 numpy로 변환 후 bytes로
        elif isinstance(value, list):
            return np.array(value, dtype=np.float32).tobytes()
        return value

    def process_result_value(self, value, dialect):
        """bytes -> numpy array (DB 조회 시)"""
        if value is None:
            return None
        # bytes를 numpy array로 변환 (768차원 SBERT 벡터 가정)
        return np.frombuffer(value, dtype=np.float32)


# ✅ 대안: JSON TEXT로 저장 (더 안전하고 디버깅 쉬움)
class JSONEncodedList(TypeDecorator):
    """
    Python list를 JSON 문자열로 TEXT에 저장
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Python list -> JSON 문자열 (DB 저장 시)"""
        if value is None:
            return None
        
        # numpy array면 list로 변환
        if isinstance(value, np.ndarray):
            value = value.tolist()
        
        # 이미 문자열이면 그대로 반환
        if isinstance(value, str):
            return value
        
        # list나 dict를 JSON 문자열로 변환
        # ✅ ensure_ascii=False로 한글 등 유니코드 문자 보존
        return json.dumps(value, ensure_ascii=False)

    def process_result_value(self, value, dialect):
        """JSON 문자열 -> Python list (DB 조회 시)"""
        if value is None:
            return None
        
        # 이미 list나 dict면 그대로 반환 (MySQL JSON 타입의 경우)
        if isinstance(value, (list, dict)):
            return value
        
        # bytes면 디코딩
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        
        # 문자열이면 JSON 파싱
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 파싱 실패: {e}")
                print(f"   값: {value[:100]}...")
                return None
        
        print(f"⚠️ 알 수 없는 타입: {type(value)}")
        return None


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

class ArticleRecommend(Base):
    __tablename__ = "article_recommend"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    status = Column(String(50), default='PENDING', index=True, nullable=False) 
    article = relationship("Article", back_populates="recommend", uselist=False)
    keywords = relationship("ArticleRecommendKeyword", back_populates="recommend")
    vector = relationship("ArticleRecommendVector", back_populates="recommend", uselist=False)

class ArticleRecommendKeyword(Base):
    __tablename__ = "article_recommend_keywords"
    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(100), index=True)
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="keywords")

class ArticleRecommendVector(Base):
    __tablename__ = "article_recommend_vector"
    id = Column(Integer, primary_key=True, index=True)
    sbert_vector = Column(JSON, nullable=False)
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="vector")
    __table_args__ = (
        UniqueConstraint('article_recommend_id', name='uq_reco_id_vector'),
    )

# 👇👇👇 [추가됨] 백엔드의 User 관련 테이블 3개 정의 👇👇👇

class User(Base):
    """
    백엔드가 관리하는 User 테이블. 
    AI 서버는 이 테이블의 id만 참조합니다.
    """
    __tablename__ = "users" # (테이블 이름이 'users'라고 가정)
    id = Column(Integer, primary_key=True, index=True)
    # (다른 컬럼들은 AI 서버가 알 필요 없음)
    
    # User가 UserCategory를 여러 개 가짐
    categories = relationship("UserCategory", back_populates="user")
    # User가 UserRecentArticle을 여러 개 가짐
    recent_articles = relationship("UserRecentArticle", back_populates="user")

class UserCategory(Base):
    """
    사용자가 선호하는 카테고리 (LLM 콜드 스타트용)
    """
    __tablename__ = "user_category"
    id = Column(Integer, primary_key=True, index=True)
    category_name = Column(String(100)) # (컬럼명이 'category_name'이라고 가정)
    
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="categories")

class UserRecentArticle(Base):
    """
    사용자가 최근 읽은 기사 (SBERT 추천용)
    """
    __tablename__ = "user_recent_article"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="recent_articles")
    
    article_id = Column(Integer, ForeignKey("article.id"))