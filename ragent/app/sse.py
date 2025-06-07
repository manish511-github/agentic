from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from .tasks import add_project_agent_connection, remove_project_agent_connection
from .models import ProjectModel, AgentModel
from .database import AsyncSessionLocal
from .auth import get_current_user
from sqlalchemy import select
import asyncio
import json
import structlog
from typing import Dict, Set, Tuple

logger = structlog.get_logger()
router = APIRouter()

def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        db.close()

async def verify_project_agent_access(db, project_id: int, agent_id: int, user_id: int) -> bool:
    """Verify if user has access to the project and agent"""
    try:
        # Check if project exists and user is the owner
        result = await db.execute(
            select(ProjectModel).filter(
                ProjectModel.id == project_id,
                ProjectModel.owner_id == user_id
            )
        )
        project = result.scalar_one_or_none()
        
        if not project:
            return False
            
        # Check if agent exists and belongs to the project
        result = await db.execute(
            select(AgentModel).filter(
                AgentModel.id == agent_id,
                AgentModel.project_id == project_id
            )
        )
        agent = result.scalar_one_or_none()
        
        return agent is not None
    except Exception as e:
        logger.error("Access verification failed", error=str(e))
        return False

@router.get("/projects/{project_id}/agents/{agent_id}/events")
async def project_agent_sse_endpoint(
    project_id: int,
    agent_id: int,
    token: str = Query(None),
    db = Depends(get_db)
):
    """SSE endpoint for project-specific agent updates with user authentication"""
    try:
        # Use the existing get_current_user function
        user = await get_current_user(token, db)
        
        # Verify user has access to the project and agent
        has_access = await verify_project_agent_access(db, project_id, agent_id, user.id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create a queue for this connection
        queue = await add_project_agent_connection(project_id, agent_id)
        
        async def event_generator():
            try:
                while True:
                    # Wait for a message
                    message = await queue.get()
                    
                    # Format the message for SSE
                    yield f"data: {json.dumps(message)}\n\n"
            except asyncio.CancelledError:
                # Clean up when the client disconnects
                await remove_project_agent_connection(project_id, agent_id, queue)
                raise
            except Exception as e:
                logger.error("Error in SSE stream", error=str(e))
                await remove_project_agent_connection(project_id, agent_id, queue)
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