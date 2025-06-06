from celery import shared_task
from sqlalchemy.orm import Session
from datetime import datetime
from .database import SessionLocal
from . import models
from app.celery_app import celery_app
from app.models import AgentModel, AgentResultModel
from app.rdagent import run_sync_workflow
from app.websocket import manager
import structlog

logger = structlog.get_logger()

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
            if agent.platform == "reddit":
                # Prepare initial state
                state = {
                    "agent_name": agent.agent_name,
                    "goals": agent.goals.split(",") if agent.goals else [],
                    "instructions": agent.instructions,
                    "description": agent.advanced_settings.get("description", ""),
                    "expectation": agent.advanced_settings.get("expectation", ""),
                    "target_audience": agent.advanced_settings.get("target_audience", ""),
                    "company_keywords": agent.advanced_settings.get("company_keywords", []),
                    "min_upvotes": agent.advanced_settings.get("min_upvotes", 0),
                    "max_age_days": agent.advanced_settings.get("max_age_days", 7),
                    "restrict_to_goal_subreddits": agent.advanced_settings.get("restrict_to_goal_subreddits", False),
                    "subreddits": [],
                    "posts": [],
                    "retries": 0,
                    "error": None,
                    "db": session
                }
                
                # Run workflow
                result = run_sync_workflow(state)
                
                if not result.get("error"):
                    # Save results
                    agent_result = AgentResultModel(
                        agent_id=agent.id,
                        status="completed",
                        result_data=result,
                        error=None
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
                        status="error",
                        result_data=None,
                        error=result["error"]
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
            else:
                logger.warning("Unsupported platform", platform=agent.platform)
                    
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
                        status="error",
                        result_data=None,
                        error=str(e)
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