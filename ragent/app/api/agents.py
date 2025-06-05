from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import uuid
from datetime import datetime

from ..database import get_db
from .. import models, schemas
from ..auth import get_current_active_user
from app.websocket import manager
from app.tasks import run_agent  # Import the Celery task

router = APIRouter(
    prefix="/agents",
    tags=["agents"]
)

def validate_uuid(uuid_str: str) -> bool:
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False

@router.post("/", response_model=schemas.Agent)
async def create_agent(
    agent: schemas.AgentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Verify project exists and belongs to user
    result = await db.execute(
        select(models.ProjectModel).filter(
            models.ProjectModel.uuid == str(agent.project_id),
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.scalars().first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )

    # Create new agent with actual project ID
    agent_data = agent.dict()
    agent_data["project_id"] = project.id
    db_agent = models.AgentModel(**agent_data)
    db.add(db_agent)
    await db.commit()
    await db.refresh(db_agent)

    # Trigger Celery task for the new agent
    try:
        # Schedule the task to run
        run_agent.delay(db_agent.id)
        
        # Update agent status to indicate task is scheduled
        db_agent.agent_status = "scheduled"
        db_agent.last_run = datetime.utcnow()
        await db.commit()
        await db.refresh(db_agent)
        
        # Send WebSocket notification
        await manager.broadcast_to_project(
            project.id,
            {
                "type": "agent_created",
                "agent_id": db_agent.id,
                "status": "scheduled",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        # Log the error but don't fail the agent creation
        print(f"Error scheduling agent task: {str(e)}")
    
    return db_agent

@router.get("/{agent_id}", response_model=schemas.Agent)
async def get_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Get agent and verify project ownership
    result = await db.execute(
        select(models.AgentModel)
        .join(models.ProjectModel)
        .filter(
            models.AgentModel.id == agent_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    agent = result.scalars().first()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have access to it"
        )
    return agent

@router.get("/project/{project_uuid}", response_model=List[schemas.Agent])
async def get_project_agents(
    project_uuid: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Validate UUID format
    if not validate_uuid(project_uuid):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project UUID format"
        )

    # Verify project ownership using UUID
    result = await db.execute(
        select(models.ProjectModel).filter(
            models.ProjectModel.uuid == project_uuid,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.scalars().first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )

    # Get all agents for the project
    result = await db.execute(
        select(models.AgentModel).filter(models.AgentModel.project_id == project.id)
    )
    agents = result.scalars().all()
    return agents

@router.put("/{agent_id}", response_model=schemas.Agent)
async def update_agent(
    agent_id: int,
    agent_update: schemas.AgentBase,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Get agent and verify project ownership
    result = await db.execute(
        select(models.AgentModel)
        .join(models.ProjectModel)
        .filter(
            models.AgentModel.id == agent_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    db_agent = result.scalars().first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have access to it"
        )

    # Update agent fields
    for field, value in agent_update.dict().items():
        setattr(db_agent, field, value)

    await db.commit()
    await db.refresh(db_agent)
    return db_agent

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Get agent and verify project ownership
    result = await db.execute(
        select(models.AgentModel)
        .join(models.ProjectModel)
        .filter(
            models.AgentModel.id == agent_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    db_agent = result.scalars().first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have access to it"
        )

    await db.delete(db_agent)
    await db.commit()
    return None

@router.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: int):
    await manager.connect(websocket, project_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)

@router.get("/{agent_id}/results", response_model=List[schemas.AgentResult])
async def get_agent_results(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Get agent and verify project ownership
    result = await db.execute(
        select(models.AgentModel)
        .join(models.ProjectModel)
        .filter(
            models.AgentModel.id == agent_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    agent = result.scalars().first()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have access to it"
        )

    # Get results
    result = await db.execute(
        select(models.AgentResultModel)
        .filter(models.AgentResultModel.agent_id == agent_id)
        .order_by(models.AgentResultModel.created_at.desc())
    )
    results = result.scalars().all()
    return results 