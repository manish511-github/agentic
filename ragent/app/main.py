from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis
import structlog
import os
from dotenv import load_dotenv
from app.agent import router as agent_router
from app.database import init_db
import math
from .hello import hello

# Initialize logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Load environment variables
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

# Initialize FastAPI app
app = FastAPI(title="Advanced Website Scraper and Reddit Marketing Agent API")

# Initialize rate limiter
@app.on_event("startup")
async def startup_event():
    redis_client = redis.from_url(REDIS_URL)
    await FastAPILimiter.init(redis_client)
    await init_db()
    hello()
    logger.info("Rate limiter addddddnd database initialized")

# Mount router
app.include_router(agent_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
