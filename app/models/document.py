from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, UniqueConstraint, LargeBinary, TypeDecorator
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import numpy as np
import json

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
    description = Column(Text, nullable=False) # (기사 본문)
    image_url = Column(String(2083))
    origin_link = Column(String(2083), nullable=False)
    publish_date = Column(DateTime(timezone=True), nullable=False)
    title = Column(Text, nullable=False)
    
    # 1:1 관계 (Article <-> ArticleDetail)
    article_detail_id = Column(Integer)
    
    # 1:1 관계 (Article <-> ArticleRecommend)
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="article", uselist=False)

    __table_args__ = (
        Index('uq_original_url_prefix', 'origin_link', unique=True, mysql_length=255),
    )


# --- 2. AI가 관리하는 'ArticleRecommend' (작업 상태) ---
class ArticleRecommend(Base):
    __tablename__ = "article_recommend"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    status = Column(String(50), default='PENDING', index=True, nullable=False) 
    
    article = relationship("Article", back_populates="recommend", uselist=False)
    keywords = relationship("ArticleRecommendKeyword", back_populates="recommend")
    vector = relationship("ArticleRecommendVector", back_populates="recommend", uselist=False)


# --- 3. AI가 관리하는 'ArticleRecommendKeyword' (LLM 키워드) ---
class ArticleRecommendKeyword(Base):
    __tablename__ = "article_recommend_keywords"
    
    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(100), index=True)
    
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="keywords")


# --- 4. AI가 관리하는 'ArticleRecommendVector' (SBERT 벡터) ---
class ArticleRecommendVector(Base):
    __tablename__ = "article_recommend_vector"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # ✅ 옵션 1: BLOB으로 저장 (빠르고 용량 적음, 바이너리 저장)
    # sbert_vector = Column(NumpyArray, nullable=False)
    
    # ✅ 옵션 2: TEXT로 JSON 저장 (디버깅 쉽고 안전, 권장)
    sbert_vector = Column(JSONEncodedList, nullable=False)
    
    article_recommend_id = Column(Integer, ForeignKey("article_recommend.id"))
    recommend = relationship("ArticleRecommend", back_populates="vector")

    __table_args__ = (
        UniqueConstraint('article_recommend_id', name='uq_reco_id_vector'),
    )