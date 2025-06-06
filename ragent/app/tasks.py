from celery import shared_task
from sqlalchemy.orm import Session
from datetime import datetime
from .database import SessionLocal
from . import models
from app.celery_app import celery_app
from app.models import AgentModel, AgentResultModel
from app.rdagent import reddit_graph
from app.websocket import manager
import structlog
import asyncio
import nest_asyncio

logger = structlog.get_logger()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

@shared_task
def run_agent_tasks():
    """Periodic task to run all active agents"""
    try:
        with SessionLocal() as session:
            agents = session.query(AgentModel).filter_by(agent_status="active").all()
            
            for agent in agents:
                # Schedule individual agent processing
                run_agent.delay(agent.id)
                
            logger.info("Scheduled agent tasks", count=len(agents))
                
    except Exception as e:
        logger.error("Failed to schedule agent tasks", error=str(e))

@shared_task
def run_agent(agent_id: int):
    """Process a single agent's tasks"""
    try:
        with SessionLocal() as session:
            # Get agent details
            agent = session.query(AgentModel).filter(AgentModel.id == agent_id).first()
            
            if not agent:
                logger.error("Agent not found", agent_id=agent_id)
                return
                
            if agent.agent_status not in ["active", "scheduled"]:
                logger.info("Agent not active or scheduled", agent_id=agent_id, status=agent.agent_status)
                return
                
            # Update agent status
            agent.agent_status = "running"
            agent.last_run = datetime.utcnow()
            session.commit()
            
            # Send WebSocket notification
            manager.broadcast_to_project(
                agent.project_id,
                {
                    "type": "agent_status",
                    "data": {
                        "agent_id": agent.id,
                        "status": "running",
                        "message": "Agent started processing"
                    }
                }
            )
            
            # Process based on platform
            if agent.agent_platform == "reddit":
                # Prepare initial state
                initial_state = {
                    "agent_name": agent.agent_name,
                    "goals": agent.goals.split(",") if agent.goals else [],
                    "instructions": agent.instructions,
                    "description": agent.project.description or "",
                    "expectation": agent.expectations or "",
                    "target_audience": agent.project.target_audience or "",
                    "company_keywords": agent.project.keywords or [],
                    "min_upvotes": agent.advanced_settings.get("min_upvotes", 0),
                    "max_age_days": agent.advanced_settings.get("max_age_days", 7),
                    "restrict_to_goal_subreddits": agent.advanced_settings.get("restrict_to_goal_subreddits", False),
                    "subreddits": [],
                    "posts": [],
                    "retries": 0,
                    "error": None,
                    "db": session
                }
                
                # Create new event loop for this task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(reddit_graph.ainvoke(initial_state))
                    logger.info("Agent execution completed", agent_id=agent_id, result=result)
                    
                    if not result.get("error"):
                        # Save results
                        agent_result = AgentResultModel(
                            agent_id=agent.id,
                            project_id=agent.project_id,
                            status="completed",
                            results=result,
                        )
                        session.add(agent_result)
                        
                        # Update agent status
                        agent.agent_status = "completed"
                        session.commit()
                        
                        # Send success notification
                        manager.broadcast_to_project(
                            agent.project_id,
                            {
                                "type": "agent_result",
                                "data": {
                                    "agent_id": agent.id,
                                    "status": "completed",
                                    "result": result
                                }
                            }
                        )
                    else:
                        # Handle error
                        agent_result = AgentResultModel(
                            agent_id=agent.id,
                            project_id=agent.project_id,
                            status="error",
                            results=None,
                        )
                        session.add(agent_result)
                        
                        # Update agent status
                        agent.agent_status = "error"
                        session.commit()
                        
                        # Send error notification
                        manager.broadcast_to_project(
                            agent.project_id,
                            {
                                "type": "agent_error",
                                "data": {
                                    "agent_id": agent.id,
                                    "error": result["error"]
                                }
                            }
                        )
                finally:
                    # Clean up the event loop
                    try:
                        # Cancel all running tasks
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        
                        # Run the loop until all tasks are cancelled
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        
                        # Stop the loop
                        loop.stop()
                        loop.close()
                    except Exception as e:
                        logger.warning("Error cleaning up event loop", error=str(e))
            else:
                logger.warning("Unsupported platform", platform=agent.agent_platform)
                    
    except Exception as e:
        logger.error("Agent processing failed", agent_id=agent_id, error=str(e))
        
        try:
            with SessionLocal() as session:
                # Get agent
                agent = session.query(AgentModel).filter(AgentModel.id == agent_id).first()
                
                if agent:
                    agent.agent_status = "error"
                    
                    # Save error result
                    agent_result = AgentResultModel(
                        agent_id=agent.id,
                        project_id=agent.project_id,
                        status="error",
                        results=None,
                    )
                    session.add(agent_result)
                    session.commit()
                    
                    # Send error notification
                    manager.broadcast_to_project(
                        agent.project_id,
                        {
                            "type": "agent_error",
                            "data": {
                                "agent_id": agent.id,
                                "error": str(e)
                            }
                        }
                    )
        except Exception as inner_e:
            logger.error("Failed to handle error state", agent_id=agent_id, error=str(inner_e)) 