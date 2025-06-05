from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from app.models import Base
import structlog

# Initialize logging
logger = structlog.get_logger()

# Load environment variables
load_dotenv()

# Get database configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragent")

# Construct DSN if not provided
POSTGRES_DSN = os.getenv("POSTGRES_DSN") or f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
SYNC_POSTGRES_DSN = os.getenv("POSTGRES_DSN") or f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# SQLAlchemy setup
engine = create_async_engine(POSTGRES_DSN, echo=False)
sync_engine = create_engine(SYNC_POSTGRES_DSN, echo=False)

# Create session makers
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
SessionLocal = sessionmaker(sync_engine, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")

# Dependency for async session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session 