from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict
import uuid

from ..database import get_db
from .. import models, schemas
from ..auth import get_current_active_user
from ..tasks.background_tasks import BackgroundTaskManager

router = APIRouter(
    prefix="/agents",
    tags=["agents"]
)

task_manager = BackgroundTaskManager()

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
    
    # Start background task
    await task_manager.start_agent_task(db_agent, db)
    
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

@router.post("/{agent_id}/start")
async def start_agent_task(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    agent = await db.get(models.AgentModel, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    await task_manager.start_agent_task(agent, db)
    return {"status": "started"}

@router.post("/{agent_id}/stop")
async def stop_agent_task(
    agent_id: int,
    current_user: models.UserModel = Depends(get_current_active_user)
):
    await task_manager.stop_agent_task(agent_id)
    return {"status": "stopped"}

@router.get("/{agent_id}/status")
async def get_agent_status(
    agent_id: int,
    current_user: models.UserModel = Depends(get_current_active_user)
):
    status = task_manager.agent_states.get(agent_id)
    if not status:
        raise HTTPException(status_code=404, detail="Agent not found")
    return status

@router.get("/running", response_model=List[Dict])
async def get_running_agents(
    current_user: models.UserModel = Depends(get_current_active_user)
):
    """Get information about all running agents"""
    return await task_manager.get_running_agents()

@router.post("/stop-all")
async def stop_all_agents(
    current_user: models.UserModel = Depends(get_current_active_user)
):
    """Stop all running agent tasks"""
    await task_manager.stop_all_tasks()
    return {"status": "all agents stopped"} 