from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, or_, cast, Integer, String
from typing import List, Dict, Optional
import uuid

from ..database import get_db
from .. import models, schemas
from ..auth3 import get_current_active_user

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
    # Create the new project in the database with explicit UUID generation
    project_data = project.dict()
    project_data["owner_id"] = current_user.id
    project_data["uuid"] = str(uuid.uuid4())  # Explicitly generate UUID as string
    
    db_project = models.ProjectModel(**project_data)
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

@router.get("/{project_identifier}", response_model=schemas.Project)
async def read_project(
    project_identifier: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Try to convert project_identifier to integer for ID comparison
    try:
        project_id = int(project_identifier)
        # If conversion succeeds, search by both ID and UUID
        result = await db.execute(
            select(models.ProjectModel)
            .filter(
                or_(
                    cast(models.ProjectModel.id, String) == project_identifier,
                    models.ProjectModel.uuid == project_identifier
                ),
                models.ProjectModel.owner_id == current_user.id
            )
        )
    except ValueError:
        # If conversion fails, search only by UUID
        result = await db.execute(
            select(models.ProjectModel)
            .filter(
                models.ProjectModel.uuid == project_identifier,
                models.ProjectModel.owner_id == current_user.id
            )
        )
    
    project = result.scalars().first()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Project not found or you do not have access to it"
        )

    return project

@router.delete("/{project_identifier}")
async def delete_project(
    project_identifier: str,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Try to convert project_identifier to integer for ID comparison
    try:
        project_id = int(project_identifier)
        # If conversion succeeds, search by both ID and UUID
        result = await db.execute(
            select(models.ProjectModel)
            .filter(
                or_(
                    models.ProjectModel.id == project_id,
                    models.ProjectModel.uuid == project_identifier
                ),
                models.ProjectModel.owner_id == current_user.id
            )
        )
    except ValueError:
        # If conversion fails, search only by UUID
        result = await db.execute(
            select(models.ProjectModel)
            .filter(
                models.ProjectModel.uuid == project_identifier,
                models.ProjectModel.owner_id == current_user.id
            )
        )
    
    project = result.scalars().first()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Project not found or you do not have access to it"
        )

    # Delete the project
    await db.delete(project)
    await db.commit()

    return {"message": f"Project deleted successfully"}

@router.put("/{project_identifier}", response_model=schemas.Project)
async def update_project(
    project_identifier: str,
    project_update: schemas.ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_active_user)
):
    # Try to convert project_identifier to integer for ID comparison
    try:
        project_id = int(project_identifier)
        # If conversion succeeds, search by both ID and UUID
        result = await db.execute(
            select(models.ProjectModel)
            .filter(
                or_(
                    models.ProjectModel.id == project_id,
                    models.ProjectModel.uuid == project_identifier
                ),
                models.ProjectModel.owner_id == current_user.id
            )
        )
    except ValueError:
        # If conversion fails, search only by UUID
        result = await db.execute(
            select(models.ProjectModel)
            .filter(
                models.ProjectModel.uuid == project_identifier,
                models.ProjectModel.owner_id == current_user.id
            )
        )
    
    project = result.scalars().first()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Project not found or you do not have access to it"
        )

    # Update project fields
    update_data = project_update.dict(exclude_unset=True)
    for field in update_data:
        setattr(project, field, update_data[field])

    await db.commit()
    await db.refresh(project)

    return project
