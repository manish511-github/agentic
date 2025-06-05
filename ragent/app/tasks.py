from celery import shared_task
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from .database import AsyncSessionLocal
from . import models
from app.celery_app import celery_app
from app.rdagent import RedditAgent
from app.models import AgentModel, AgentResultModel
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.websocket import manager
import logging
import asyncio
import json
import structlog

logger = structlog.get_logger()

@shared_task
def run_agent_tasks():
    """Periodic task to run all active agents"""
    try:
        # Get all active agents
        with SessionLocal() as session:
            agents = session.query(AgentModel).filter_by(status="active").all()
            
            for agent in agents:
                # Schedule individual agent processing
                run_agent.delay(agent.id)
                
            logger.info("Scheduled agent tasks", count=len(agents))
            
    except Exception as e:
        logger.error("Failed to schedule agent tasks", error=str(e))

@shared_task
def process_agent(agent_id: int):
    """Process a single agent's tasks"""
    async def _process():
        async with AsyncSessionLocal() as session:
            # Get agent details
            result = await session.execute(
                select(models.AgentModel).filter(
                    models.AgentModel.id == agent_id
                )
            )
            agent = result.scalars().first()
            
            if not agent:
                return
            
            # Process based on platform
            if agent.agent_platform == "reddit":
                await process_reddit_agent(agent, session)
            elif agent.agent_platform == "twitter":
                await process_twitter_agent(agent, session)
            elif agent.agent_platform == "linkedin":
                await process_linkedin_agent(agent, session)
    
    # Run the async function
    asyncio.run(_process())

async def process_reddit_agent(agent: models.AgentModel, session: AsyncSession):
    """Process Reddit-specific agent tasks"""
    # Implement Reddit-specific logic here
    pass

async def process_twitter_agent(agent: models.AgentModel, session: AsyncSession):
    """Process Twitter-specific agent tasks"""
    # Implement Twitter-specific logic here
    pass

async def process_linkedin_agent(agent: models.AgentModel, session: AsyncSession):
    """Process LinkedIn-specific agent tasks"""
    # Implement LinkedIn-specific logic here
    pass

@shared_task
def run_agent(agent_id: int):
    """Process a single agent's tasks"""
    # Create event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Get agent from database
        with SessionLocal() as session:
            agent = session.query(AgentModel).get(agent_id)
            if not agent:
                logger.error("Agent not found", agent_id=agent_id)
                return
                
            if agent.status not in ["active", "scheduled"]:
                logger.info("Agent not active or scheduled", agent_id=agent_id, status=agent.status)
                return
                
            # Update agent status
            agent.status = "running"
            agent.last_run = datetime.utcnow()
            session.commit()
            
            # Send WebSocket notification
            loop.run_until_complete(manager.broadcast_to_project(
                agent.project_id,
                {
                    "type": "agent_status",
                    "data": {
                        "agent_id": agent.id,
                        "status": "running",
                        "message": "Agent started processing"
                    }
                }
            ))
            
            # Process based on platform
            if agent.platform == "reddit":
                # Create database session
                async def process_agent():
                    async with AsyncSessionLocal() as session:
                        # Create and run Reddit agent
                        reddit_agent = RedditAgent(agent, session)
                        return await reddit_agent.run()
                
                # Run the agent
                result = loop.run_until_complete(process_agent())
                
                if result["status"] == "completed":
                    # Save results
                    agent_result = AgentResultModel(
                        agent_id=agent.id,
                        status="completed",
                        result_data=result,
                        error=None
                    )
                    session.add(agent_result)
                    
                    # Update agent status
                    agent.status = "completed"
                    session.commit()
                    
                    # Send success notification
                    loop.run_until_complete(manager.broadcast_to_project(
                        agent.project_id,
                        {
                            "type": "agent_result",
                            "data": {
                                "agent_id": agent.id,
                                "status": "completed",
                                "result": result
                            }
                        }
                    ))
                else:
                    # Handle error
                    agent_result = AgentResultModel(
                        agent_id=agent.id,
                        status="error",
                        result_data=None,
                        error=result["error"]
                    )
                    session.add(agent_result)
                    
                    # Update agent status
                    agent.status = "error"
                    session.commit()
                    
                    # Send error notification
                    loop.run_until_complete(manager.broadcast_to_project(
                        agent.project_id,
                        {
                            "type": "agent_error",
                            "data": {
                                "agent_id": agent.id,
                                "error": result["error"]
                            }
                        }
                    ))
            else:
                logger.warning("Unsupported platform", platform=agent.platform)
                
    except Exception as e:
        logger.error("Agent processing failed", agent_id=agent_id, error=str(e))
        
        with SessionLocal() as session:
            # Update agent status
            agent = session.query(AgentModel).get(agent_id)
            if agent:
                agent.status = "error"
                
                # Save error result
                agent_result = AgentResultModel(
                    agent_id=agent.id,
                    status="error",
                    result_data=None,
                    error=str(e)
                )
                session.add(agent_result)
                session.commit()
                
                # Send error notification
                loop.run_until_complete(manager.broadcast_to_project(
                    agent.project_id,
                    {
                        "type": "agent_error",
                        "data": {
                            "agent_id": agent.id,
                            "error": str(e)
                        }
                    }
                ))
    finally:
        loop.close() 