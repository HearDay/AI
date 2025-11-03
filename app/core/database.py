from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# 프로젝트 폴더에 'myapi.db' 라는 이름의 SQLite 데이터베이스 파일이 생성됩니다.
DATABASE_URL = "sqlite+aiosqlite:///./myapi.db"

# 데이터베이스와 통신하는 '엔진'을 만듭니다.
engine = create_async_engine(DATABASE_URL)

# 데이터베이스와 대화(세션)를 나누기 위한 창구를 만듭니다.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession
)

# 우리가 만들 테이블(모델)들이 상속받을 기본 클래스입니다.
Base = declarative_base()

# API가 요청될 때마다 DB 세션을 생성하고, 끝나면 닫아주는 함수입니다.
async def get_db():
    async with SessionLocal() as session:
        yield session