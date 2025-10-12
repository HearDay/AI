from sqlalchemy import Column, Integer, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    keywords = Column(JSON)      # LLM 키워드를 저장할 컬럼 (원래 계획대로)
    sbert_vector = Column(JSON)  # SBERT 벡터를 저장할 컬럼 (새로 추가)
    created_at = Column(DateTime(timezone=True), server_default=func.now())