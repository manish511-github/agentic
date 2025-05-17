from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from app.models import Base
import structlog

# Initialize logging
logger = structlog.get_logger()

# Load environment variables
load_dotenv()
POSTGRES_DSN = os.getenv("POSTGRES_DSN")

# SQLAlchemy setup
engine = create_async_engine(POSTGRES_DSN, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")

# Dependency for async session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session 