from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from pydantic_settings import BaseSettings
from pydantic import Field

# 1. .env 파일을 읽어올 설정 클래스
class DatabaseSettings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    class Config:
        env_file = ".env" # .env 파일을 읽어들임

# 설정 인스턴스 생성
settings = DatabaseSettings()

# MySQL 비동기 연결 URL
DATABASE_URL = (
    f"mysql+asyncmy://{settings.DB_USER}:{settings.DB_PASSWORD}@"
    f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    f"?charset=utf8mb4"
)

# 3. 비동기 엔진 생성
engine = create_async_engine(
    DATABASE_URL, 
    echo=True, 
    pool_pre_ping=True,
    pool_recycle=3600,  # 1시간마다 커넥션 재생성 (MySQL 타임아웃 방지)
)

# ✅ async_sessionmaker 사용 (sessionmaker 대신)
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # 백그라운드 작업에서 객체 재사용 가능
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()

# Dependency for FastAPI
async def get_db():
    """
    FastAPI 의존성으로 사용할 DB 세션 생성기
    """
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()