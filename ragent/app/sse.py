from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from .tasks import add_project_agent_connection, remove_project_agent_connection, send_project_agent_update
from .models import ProjectModel, AgentModel
from .database import AsyncSessionLocal
from .auth3 import get_current_user
from sqlalchemy import select
import asyncio
import json
import structlog
from typing import Dict, Set, Tuple, AsyncGenerator
from uuid import UUID
from datetime import datetime
import redis

logger = structlog.get_logger()
router = APIRouter()

# Use the Redis URL with DB 1 for SSE messages
REDIS_URL = "redis://redis:6379/1"  # Using DB 1 for SSE

# Initialize Redis client
redis_client = redis.from_url(REDIS_URL)

async def get_db() -> AsyncGenerator:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def verify_project_access(db, project_id: UUID, user_id: int) -> bool:
    """Verify if user has access to the project"""
    try:
        # Convert the UUID to string for comparison with the VARCHAR column
        project_id_str = str(project_id)
        
        result = await db.execute(
            select(ProjectModel).filter(
                ProjectModel.uuid == project_id_str,  # Compare with string
                ProjectModel.owner_id == user_id
            )
        )
        project = result.scalar_one_or_none()
        return project is not None
    except Exception as e:
        logger.error("Access verification failed", error=str(e))
        return False

@router.post("/projects/{project_id}/test-event", tags=["sse"])
async def test_event(
    project_id: UUID,
    db = Depends(get_db)
):
    """Send a test event to all connected clients for a project"""
    try:
        # Skip authentication and directly send the message
        await send_project_agent_update(
            str(project_id),
            None,
            {
                "type": "test_event",
                "data": {
                    "project_id": str(project_id),
                    "message": "Test message from server",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        return {"status": "success"}
    except Exception as e:
        logger.error("Failed to send test message", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to send test message")

@router.get("/projects/{project_id}/events", tags=["sse"])
async def project_sse_endpoint(
    project_id: UUID,
    token: str = Query(None),
    db = Depends(get_db)
):
    """SSE endpoint for project-specific updates with user authentication"""
    try:
        # Use the existing get_current_user function
        user = await get_current_user(token, db)
        
        # Verify user has access to the project
        has_access = await verify_project_access(db, project_id, user.id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create Redis pubsub instance
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"sse:project:{project_id}")
        
        async def event_generator():
            try:
                while True:
                    # Get message from Redis using asyncio's event loop
                    message = await asyncio.get_event_loop().run_in_executor(
                        None, pubsub.get_message
                    )
                    if message and message['type'] == 'message':
                        data = json.loads(message['data'])
                        yield f"data: {json.dumps(data)}\n\n"
                    else:
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Clean up when the client disconnects
                pubsub.unsubscribe(f"sse:project:{project_id}")
                raise
            except Exception as e:
                logger.error("Error in SSE stream", error=str(e))
                pubsub.unsubscribe(f"sse:project:{project_id}")
                raise
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error("Failed to create SSE connection", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create SSE connection") 