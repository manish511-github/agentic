from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List
import uuid
from datetime import datetime, timezone
from pydantic import ValidationError
import structlog

from ...database import get_db
from ... import models, schemas
from app.auth.security import get_current_user, get_jwt_identity
from app.websocket import manager
from app.tasks import run_agent  # Import the Celery task

# Initialize logging
import logging

# Standard logging configuration
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)

# Structlog configuration
structlog.configure(
    processors=[
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

# Initialize logger
logger = structlog.get_logger()

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
    agent_data: schemas.AgentCreate,  # Accept raw dictionary
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_user)
):
    # Ensure agent_platform is present
    agent_platform = agent_data.agent_platform
    if not agent_platform:
        raise HTTPException(
            status_code=400, detail="agent_platform is required")

    # Validate and parse agent data based on platform
    # Define a map to your platform-specific schemas
    schema_map = {
        "reddit": schemas.AgentCreate,
        # "twitter": schemas.TwitterAgentRequestSchema, # Add when available
    }
    schema_cls = schema_map.get(agent_platform)
    if not schema_cls:
        raise HTTPException(
            status_code=400, detail=f"Unsupported agent_platform: {agent_platform}")

    # Since FastAPI already validated the data, we can use it directly
    agent_payload = agent_data.model_dump()

    # Verify project exists and belongs to user
    project_uuid = agent_payload.get("project_id")
    result = await db.execute(
        select(models.ProjectModel).filter(
            models.ProjectModel.uuid == str(project_uuid),
            models.ProjectModel.owner_id == current_user.id
        )
    )
    project = result.scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )

    # Map validated data to AgentModel
    db_agent = models.AgentModel(
        agent_name=agent_payload["agent_name"],
        description=agent_payload.get("description"),
        agent_platform=agent_payload["agent_platform"],
        agent_status=agent_payload.get("agent_status", "active"),
        goals=agent_payload["goals"],
        instructions=agent_payload["instructions"],
        expectations=agent_payload.get("expectations", ""),
        agent_keywords=agent_payload.get("agent_keywords", []),
        project_id=project.uuid,
        mode=agent_payload["mode"],
        review_minutes=agent_payload.get("review_minutes"),
        oauth_account_id=agent_payload.get("oauth_account_id"),
        advanced_settings=agent_payload.get("advanced_settings"),
        platform_settings=agent_payload.get("platform_settings")
    )
    
    if agent_payload.get("schedule"):
        schedule_data = agent_payload["schedule"]
        schedule_type = schedule_data["schedule_type"]
        schedule_time = schedule_data.get("schedule_time")
        
        if schedule_time is not None:
            # convert schedule time string to datetime object
            schedule_time = datetime.strptime(
                schedule_time, "%Y-%m-%d %H:%M:%S")
        
        # Set default values based on schedule type if not provided
        days_of_week = None
        day_of_month = None
        
        if schedule_type == "weekly":
            # If weekly, set today's weekday name as default
            today_weekday = datetime.now(timezone.utc).weekday()
            weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            days_of_week = schedule_data.get("days_of_week", [weekday_names[today_weekday]])
        elif schedule_type == "monthly":
            # If monthly, set today's day of month as default
            today_day = datetime.now(timezone.utc).day
            day_of_month = schedule_data.get("day_of_month", today_day)
        
        db_agent.schedule = models.ScheduleModel(
            schedule_type=schedule_type,
            schedule_time=schedule_time if schedule_time is not None else datetime.now(timezone.utc).replace(tzinfo=None),
            days_of_week=days_of_week,
            day_of_month=day_of_month
        )

    db.add(db_agent)

    # Flush to assign IDs before creating related execution entries
    await db.flush()

    if agent_payload.get("schedule"):
        # Create an execution entry now that schedule and agent have IDs
        # Ensure schedule_time is never null - use current time as default if not provided
        execution_schedule_time = schedule_time if schedule_time is not None else datetime.now(timezone.utc).replace(tzinfo=None)
        
        execution = models.ExecutionModel(
            schedule_id=db_agent.schedule.id,
            agent_id=db_agent.id,
            status=models.ExecutionStatusEnum.scheduled,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            schedule_time=execution_schedule_time
        )
        db.add(execution)

    await db.commit()
    await db.refresh(db_agent)

    # Use the Agent schema for the response
    response_data = schemas.Agent.model_validate(db_agent)
    return response_data


@router.get("/{agent_id}", response_model=schemas.Agent)
async def get_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_user)
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
    current_user: models.UserModel = Depends(get_current_user)
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
        select(models.AgentModel).filter(
            models.AgentModel.project_id == project.uuid)
    )
    agents = result.scalars().all()
    return agents


@router.put("/{agent_id}", response_model=schemas.Agent)
async def update_agent(
    agent_id: int,
    agent_update: schemas.AgentBase,
    db: AsyncSession = Depends(get_db),
    current_user: models.UserModel = Depends(get_current_user)
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
    current_user: models.UserModel = Depends(get_current_user)
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
    current_user: models.UserModel = Depends(get_current_user)
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

    # Base result structure
    result_data = {
        "agent_platform": agent.agent_platform,
        "posts": []
    }

    # Handle platform-specific post retrieval
    if agent.agent_platform == "reddit":
        # Get all Reddit analysis results for this agent with post data
        # Simple approach: get all records and deduplicate in Python
        
        # Get all mapper records for this agent with post data
        posts_result = await db.execute(
            select(
                models.RedditAgentExecutionMapperModel,
                models.RedditPostModel
            )
            .join(
                models.RedditPostModel,
                models.RedditAgentExecutionMapperModel.post_id == models.RedditPostModel.post_id
            )
            .filter(models.RedditAgentExecutionMapperModel.agent_id == agent_id)
            .order_by(
                models.RedditAgentExecutionMapperModel.post_id,
                desc(models.RedditAgentExecutionMapperModel.created_at)
            )
        )
        
        all_execution_posts = posts_result.all()
        
        # Deduplicate by post_id, keeping the latest execution for each post
        seen_posts = {}
        for row in all_execution_posts:
            mapper_data = row[0]  # First element is RedditAgentExecutionMapperModel
            post_data = row[1]    # Second element is RedditPostModel
            
            if post_data.post_id not in seen_posts:
                seen_posts[post_data.post_id] = (mapper_data, post_data)
        
        # Convert to list and sort by combined_relevance
        unique_posts = list(seen_posts.values())
        unique_posts.sort(key=lambda x: x[0].combined_relevance or 0, reverse=True)
        
        if unique_posts:
            result_data["posts"] = [
                {
                    # Post data from RedditPostModel
                    "subreddit": post_data.subreddit,
                    "post_id": post_data.post_id,
                    "post_title": post_data.post_title,
                    "post_body": post_data.post_body,
                    "post_url": post_data.post_url,
                    "upvotes": post_data.upvotes,
                    "comment_count": post_data.comment_count,
                    "created_utc": post_data.created_utc.isoformat() if post_data.created_utc else None,
                    
                    # Analysis results from RedditAgentExecutionMapperModel (latest execution)
                    "execution_id": mapper_data.execution_id,
                    "relevance_score": mapper_data.relevance_score,
                    "keyword_relevance": mapper_data.keyword_relevance,
                    "semantic_relevance": mapper_data.semantic_relevance,
                    "combined_relevance": mapper_data.combined_relevance,
                    "llm_relevance": mapper_data.llm_relevance,
                    "matched_query": mapper_data.matched_query,
                    "sort_method": mapper_data.sort_method,
                    "comment_draft": mapper_data.comment_draft,
                    "status": mapper_data.status,
                    "analysis_created_at": mapper_data.created_at.isoformat() if mapper_data.created_at else None
                }
                for mapper_data, post_data in unique_posts
            ]
    
    elif agent.agent_platform == "twitter":
        # Get Twitter posts for this agent (no mapper pattern yet)
        posts_result = await db.execute(
            select(models.TwitterPostModel)
            .filter(models.TwitterPostModel.agent_name == agent.agent_name)
            .order_by(models.TwitterPostModel.created.desc())
        )
        posts = posts_result.scalars().all()

        if posts:
            result_data["posts"] = [
                {
                    "tweet_id": post.tweet_id,
                    "text": post.text,
                    "user_name": post.user_name,
                    "user_screen_name": post.user_screen_name,
                    "retweet_count": post.retweet_count,
                    "favorite_count": post.favorite_count,
                    "relevance_score": post.relevance_score,
                    "hashtags": post.hashtags,
                    "created_at": post.created.isoformat() if post.created else None
                }
                for post in posts
            ]

    # Create a single AgentResult response with all posts
    agent_result = {
        "id": agent_id,  # Use agent_id as the result id
        "agent_id": agent_id,
        "project_id": agent.project_id,
        "status": "completed" if result_data["posts"] else "no_results",
        "results": result_data,
        "error": None,
        "created_at": datetime.now(timezone.utc)  # Current timestamp
    }
    
    return [schemas.AgentResult.model_validate(agent_result)]
