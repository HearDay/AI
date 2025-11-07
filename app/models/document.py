from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Index
from sqlalchemy.sql import func
from app.core.database import Base

# 백엔드의 ArticleDetail은 복잡하니, 일단 ID만 매핑합니다.
# from sqlalchemy import ForeignKey
# from sqlalchemy.orm import relationship

class Document(Base):
    __tablename__ = "article " 

    # --- 백엔드 팀이 정의한 컬럼 ---
    # 백엔드의 'id' (Long) -> 파이썬의 'id' (Integer)
    id = Column(Integer, primary_key=True, index=True)
    
    # 'origin_link' (DB) -> 'original_url' (Python)
    original_url = Column(String(2083), name="origin_link", nullable=False)
    
    # 'publish_data' (DB) -> 'published_at' (Python)
    # (참고: Java 코드에서는 publishDate였는데, 목록에서는 publish_data네요.
    #  DB 컬럼명인 publish_data로 매핑합니다.)
    published_at = Column(DateTime(timezone=True), name="publish_data", nullable=False)
    
    # 'title' (DB) -> 'title' (Python)
    title = Column(Text, nullable=False)
    
    # 'description' (DB) -> 'text' (Python)
    # (우리가 text로 사용하던 본문입니다)
    text = Column(Text, name="description", nullable=False)
    
    # 'article_category' (DB) -> 'category' (Python)
    category = Column(String(100), name="article_category", index=True)
    
    # 매핑할 다른 컬럼들
    image_url = Column(String(2083))
    article_detail_id = Column(Integer) # (일단 Integer로 매핑)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # --- AI 서버가 추가할 컬럼 (★백엔드 팀이 DB에 추가해야 함★) ---
    keywords = Column(JSON)
    sbert_vector = Column(JSON)
    status = Column(String(50), default='PENDING', index=True, nullable=False)

    # --- 인덱스 설정 ---
    __table_args__ = (
        # 'origin_link' 컬럼을 기준으로 고유 인덱스 생성
        Index('uq_origin_link_prefix', 'origin_link', unique=True, mysql_length=255),
    )