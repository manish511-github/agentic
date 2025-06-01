from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete # Import select and delete for queries
from typing import List, Dict, Optional

from ..database import get_db
from .. import models, schemas
from ..auth import get_current_active_user

router = APIRouter(
    prefix="/projects",
    tags=["projects"]
)

@router.post("/", response_model=schemas.Project)
async def create_project(
    project: schemas.ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Create the new project in the database
    db_project = models.ProjectModel(**project.dict(), owner_id=current_user.id)
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    return db_project

@router.get("/", response_model=List[schemas.Project])
async def read_projects(
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Retrieve projects for the current user
    result = await db.execute(
        select(models.ProjectModel).filter(models.ProjectModel.owner_id == current_user.id)
    )
    projects = result.scalars().all()
    return projects

@router.get("/{project_id}", response_model=schemas.Project)
async def read_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Retrieve a specific project by ID for the current user
    result = await db.execute(
        select(models.ProjectModel)
        .filter(
            models.ProjectModel.id == project_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.scalars().first()

    if project is None:
        raise HTTPException(status_code=404, detail="Project not found or you do not have access to it")

    return project

@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Find the project to delete, ensuring it belongs to the current user
    result = await db.execute(
        select(models.ProjectModel)
        .filter(
            models.ProjectModel.id == project_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.scalars().first()

    if project is None:
        raise HTTPException(status_code=404, detail="Project not found or you do not have access to it")

    # Delete the project
    await db.delete(project)
    await db.commit()

    return {"message": f"Project with id {project_id} deleted successfully"}

@router.put("/{project_id}", response_model=schemas.Project)
async def update_project(
    project_id: int,
    project_update: schemas.ProjectCreate, # Use the create schema for update data
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Find the project to update, ensuring it belongs to the current user
    result = await db.execute(
        select(models.ProjectModel)
        .filter(
            models.ProjectModel.id == project_id,
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.scalars().first()

    if project is None:
        raise HTTPException(status_code=404, detail="Project not found or you do not have access to it")

    # Update project fields
    update_data = project_update.dict(exclude_unset=True) # Use exclude_unset to only update provided fields
    for field in update_data:
        setattr(project, field, update_data[field])

    await db.commit()
    await db.refresh(project)

    return project
