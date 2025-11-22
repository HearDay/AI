from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic_settings import BaseSettings

# 1. .env 파일을 읽어올 설정 클래스
class DatabaseSettings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    class Config:
        env_file = ".env"

settings = DatabaseSettings()

# --- 1. 비동기 엔진 (FastAPI API용) ---
# greenlet 기반의 'asyncmy' 드라이버 사용 (API 응답용)
ASYNC_DATABASE_URL = (
    f"mysql+asyncmy://{settings.DB_USER}:{settings.DB_PASSWORD}@"
    f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    f"?charset=utf8mb4"
)
engine = create_async_engine(ASYNC_DATABASE_URL, echo=True, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine, 
    class_=AsyncSession
)

# --- 2. 동기 엔진 (백그라운드 작업용) ---
# [필수] greenlet을 쓰지 않는 'pymysql' 드라이버 사용 (백그라운드 AI 작업용)
# 이 부분이 없어서 ImportError가 발생했습니다.
SYNC_DATABASE_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
    f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    f"?charset=utf8mb4"
)
sync_engine = create_engine(SYNC_DATABASE_URL, echo=True, pool_pre_ping=True)

# 동기 세션 팩토리 (이름: SessionLocalSync)
SessionLocalSync = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=sync_engine
)

# --- 3. 공통 Base ---
Base = declarative_base()

# 비동기 세션 주입 (API용 Dependency)
async def get_db():
    async with SessionLocal() as session:
        yield session