from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
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

# 2. SQLAlchemy에 맞는 MySQL 연결 URL 생성
DATABASE_URL = (
    f"mysql+aiomysql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
    f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    f"?charset=utf8mb4" # UTF-8 설정을 위해
)

# 3. 비동기 엔진 및 세션 생성 (이 부분은 거의 동일)
engine = create_async_engine(DATABASE_URL, echo=True, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine, 
    class_=AsyncSession
)

Base = declarative_base()

async def get_db():
    async with SessionLocal() as session:
        yield session