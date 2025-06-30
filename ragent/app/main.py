from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis
import structlog
import os
from dotenv import load_dotenv
from app.settings import get_settings
from app.agent import router as agent_router
from app.core.agents_tasks.rdagent.rd_agent_v1.router import router as rdagent_router
from app.core.agents_tasks.xagent.api_routes import router as xdagent_router
# from app.auth3 import router as auth_router
from app.api.projects import router as projects_router
from app.api.agents import router as agents_router  # Import the agents router
from app.sse import router as sse_router  # Add this import
# Import the agent generator router
from app.api.generate_profile import router as agent_generator_router
from app.database import init_db
from app.core.agents_tasks.hn_agent.hnagent import router as hnagent_router
# Import the xagent router
from app.core.agents_tasks.xagent import router as xagent_router
from app.api.auth.users import user_router
from app.api.auth.users import guest_router
from app.api.auth.users import auth_router
from app.api.auth.google_auth import google_auth_router

#Auth2 dependency
from starlette.middleware.sessions import SessionMiddleware

import math
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware

# Initialize logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

settings = get_settings()
logger = structlog.get_logger()

# Load environment variables
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

# Initialize FastAPI app
app = FastAPI(title="Advanced Website Scraper and Reddit Marketing Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://rdagent-frontend.vercel.app"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
# Initialize rate limiter


@app.on_event("startup")
async def startup_event():
    redis_client = redis.from_url(REDIS_URL)
    await FastAPILimiter.init(redis_client)
    await init_db()
    logger.info("Rate limiter and database initialized")

# Mount router
app.include_router(agent_router)
app.include_router(rdagent_router)
# app.include_router(auth_router, prefix="/auth",
#                    tags=["auth"])  # Include the auth router
app.include_router(projects_router)
app.include_router(agents_router)  # Include the agents router
app.include_router(sse_router, prefix="/sse", tags=["sse"])  # Add this line
app.include_router(agent_generator_router, prefix="/agents",
                   tags=["agent"])  # Include the agent generator router
app.include_router(hnagent_router, tags=["hackernews"])
app.include_router(xagent_router, tags=["xagent"])  # Include the xagent router

# Authentication routes
app.include_router(user_router) # new auth user router
app.include_router(guest_router)
app.include_router(auth_router)
app.include_router(google_auth_router)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
