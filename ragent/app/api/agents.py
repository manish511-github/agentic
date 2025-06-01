from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from ..database import get_db
from .. import models, schemas
from ..auth import get_current_active_user

router = APIRouter(
    prefix="/agents",
    tags=["agents"]
)

@router.post("/", response_model=schemas.Agent)
async def create_agent(
    agent: schemas.AgentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Verify project exists and belongs to user
    result = await db.execute(
        select(models.ProjectModel).filter(
            models.ProjectModel.id == agent.project_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )

    # Create new agent
    db_agent = models.AgentModel(**agent.dict())
    db.add(db_agent)
    await db.commit()
    await db.refresh(db_agent)
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
    agent = result.first()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have access to it"
        )
    return agent

@router.get("/{project_id}", response_model=List[schemas.Agent])
async def get_project_agents(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Verify project ownership
    result = await db.execute(
        select(models.ProjectModel).filter(
            models.ProjectModel.id == project_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )

    # Get all agents for the project
    result = await db.execute(
        select(models.AgentModel).filter(models.AgentModel.project_id == project_id)
    )
    agents = result.all()
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
    db_agent = result.first()
    
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
    db_agent = result.first()
    
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have access to it"
        )

    await db.delete(db_agent)
    await db.commit()
    return None 