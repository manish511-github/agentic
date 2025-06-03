from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis
import structlog
import os
from dotenv import load_dotenv
from app.agent import router as agent_router
from app.rdagent import router as rdagent_router
from app.auth import router as auth_router 
from app.api.projects import router as projects_router 
from app.api.agents import router as agents_router
from app.api.websocket import router as websocket_router
from app.database import init_db
from fastapi.middleware.cors import CORSMiddleware
from app.tasks.background_tasks import task_manager

import math

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize rate limiter
@app.on_event("startup")
async def startup_event():
    redis_client = redis.from_url(REDIS_URL)
    await FastAPILimiter.init(redis_client)
    await init_db()
    logger.info("Rate limiter and database initialized")

# Mount routers
app.include_router(agent_router)
app.include_router(rdagent_router)
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(projects_router)
app.include_router(agents_router)
app.include_router(websocket_router)  # Add WebSocket router

@app.on_event("shutdown")
async def shutdown_event():
    # Stop all running tasks
    for agent_id in list(task_manager.running_tasks.keys()):
        await task_manager.stop_agent_task(agent_id)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
