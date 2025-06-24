from celery import shared_task
from sqlalchemy.orm import Session
from datetime import datetime
from .database import SessionLocal
from . import models
from app.celery_app import celery_app
from app.models import AgentModel, AgentResultModel, RedditPostModel, ProjectModel, TwitterPostModel
from app.core.agents_tasks.rdagent.rd_agent_advanced.graph import reddit_graph
import structlog
import asyncio
import nest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
import os
import json
import redis
from typing import Dict, Set, Tuple, Optional

logger = structlog.get_logger()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Create async engine and session factory for the workflow
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/ragent")
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Store active SSE connections for each project-agent pair
project_agent_connections: Dict[Tuple[str, Optional[int]], Set[asyncio.Queue]] = {}

# Use the Redis URL with DB 1 for SSE messages
REDIS_URL = "redis://redis:6379/1"  # Using DB 1 for SSE

# Initialize Redis client using DB 1
redis_client = redis.from_url(REDIS_URL)

async def add_project_agent_connection(project_id: str, agent_id: Optional[int] = None) -> asyncio.Queue:
    """Add a new SSE connection for a project or project-agent pair"""
    key = (project_id, agent_id)
    if key not in project_agent_connections:
        project_agent_connections[key] = set()
    
    queue = asyncio.Queue()
    project_agent_connections[key].add(queue)
    return queue

async def remove_project_agent_connection(project_id: str, queue: asyncio.Queue, agent_id: Optional[int] = None):
    """Remove an SSE connection for a project or project-agent pair"""
    key = (project_id, agent_id)
    if key in project_agent_connections:
        project_agent_connections[key].discard(queue)
        if not project_agent_connections[key]:
            del project_agent_connections[key]

def send_project_agent_update(project_id: str, agent_id: Optional[int], message: dict):
    """Send an update to all connections for a project or specific project-agent pair"""
    try:
        logger.info("Sending SSE message", 
                   project_id=project_id, 
                   agent_id=agent_id, 
                   message=message)
        
        # Publish message to Redis channel
        redis_client.publish(
            f"sse:project:{project_id}",
            json.dumps(message)
        )
        
        logger.info("Successfully sent message", 
                   project_id=project_id)
    except Exception as e:
        logger.error("Failed to send SSE message", 
                    error=str(e), 
                    project_id=project_id)

def run_async(coro):
    """Helper function to run async code in sync context"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error("Error in run_async", error=str(e), exc_info=True)
        raise

def extract_storable_data(state):
    """Extract only the necessary data from state that can be stored"""
    try:
        # If there's an error, just return the error message
        if state.get("error"):
            return {"error": str(state["error"])}
            
        # Create a new dictionary with only the data we need
        storable_data = {
            "agent_name": state.get("agent_name"),
            "goals": state.get("goals", []),
            "instructions": state.get("instructions"),
            "description": state.get("description"),
            "expectation": state.get("expectation"),
            "target_audience": state.get("target_audience"),
            "company_keywords": state.get("company_keywords", []),
            "min_upvotes": state.get("min_upvotes", 0),
            "max_age_days": state.get("max_age_days", 7),
            "restrict_to_goal_subreddits": state.get("restrict_to_goal_subreddits", False),
            "subreddits": state.get("subreddits", []),
            "posts": [
                {
                    "subreddit": post.get("subreddit"),
                    "post_id": post.get("post_id"),
                    "post_title": post.get("post_title"),
                    "post_body": post.get("post_body"),
                    "post_url": post.get("post_url"),
                    "upvotes": post.get("upvotes", 0),
                    "comment_count": post.get("comment_count", 0),
                    "created": post.get("created"),
                    "keyword_relevance": post.get("keyword_relevance"),
                    "matched_query": post.get("matched_query"),
                    "semantic_relevance": post.get("semantic_relevance"),
                    "combined_relevance": post.get("combined_relevance")
                }
                for post in state.get("posts", [])
            ],
            "retries": state.get("retries", 0)
        }
        
        # Remove any None values to keep the data clean
        return {k: v for k, v in storable_data.items() if v is not None}
    except Exception as e:
        # If anything goes wrong during extraction, return the error
        return {"error": str(e)}

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
                logger.info("Agent not active or scheduled", 
                          agent_id=agent_id, 
                          status=agent.agent_status)
                return
                
            # Update agent status
            agent.agent_status = "running"
            agent.last_run = datetime.utcnow()
            session.commit()

            # Get the project UUID
            project = session.query(ProjectModel).filter(ProjectModel.id == agent.project_id).first()
            if not project:
                logger.error("Project not found", project_id=agent.project_id)
                return

            project_uuid = str(project.uuid)
            logger.info("Sending ready status via SSE", 
                       agent_id=agent_id, 
                       project_id=project_uuid)

            # Send ready status via Redis
            send_project_agent_update(
                project_id=project_uuid,
                agent_id=agent.id,
                message={
                    "type": "agent_status",
                    "data": {
                        "project_id": project_uuid,
                        "agent_id": agent.id,
                        "status": "ready",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            )
            
            logger.info("Successfully sent ready status", 
                       agent_id=agent_id, 
                       project_id=project_uuid)
            
            # Process based on platform
            if agent.agent_platform == "reddit":
                # Create new event loop for this task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Send processing status
                    send_project_agent_update(
                        project_id=project_uuid,
                        agent_id=agent.id,
                        message={
                            "type": "agent_status",
                            "data": {
                                "project_id": project_uuid,
                                "agent_id": agent.id,
                                "status": "processing",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    )

                    # Create async session for the workflow
                    async def run_workflow():
                        async with async_session() as async_db:
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
                                "db": async_db
                            }
                            
                            result = await reddit_graph.ainvoke(initial_state)
                            return result
                    
                    # Run the workflow
                    result = loop.run_until_complete(run_workflow())
                    logger.info("Agent execution completed", agent_id=agent_id, result=result)
                    
                    # Extract storable data from result
                    storable_data = extract_storable_data(result)
                    
                    if not storable_data.get("error"):
                        # Store Reddit posts
                        for post in result.get("posts", []):
                            reddit_post = RedditPostModel(
                                agent_name=agent.agent_name,
                                goals=agent.goals.split(",") if agent.goals else [],
                                instructions=agent.instructions,
                                subreddit=post["subreddit"],
                                post_id=post["post_id"],
                                post_title=post["post_title"],
                                post_body=post["post_body"],
                                post_url=post["post_url"],
                                upvotes=post.get("upvotes", 0),
                                comment_count=post.get("comment_count", 0),
                                created=datetime.fromisoformat(post["created"]) if post.get("created") else None,
                                keyword_relevance=post.get("keyword_relevance"),
                                matched_query=post.get("matched_query"),
                                semantic_relevance=post.get("semantic_relevance"),
                                combined_relevance=post.get("combined_relevance")
                            )
                            session.add(reddit_post)
                        
                        # Save agent results
                        agent_result = AgentResultModel(
                            agent_id=agent.id,
                            project_id=agent.project_id,
                            status="completed",
                            results=storable_data,
                        )
                        session.add(agent_result)
                        
                        # Update agent status
                        agent.agent_status = "completed"
                        session.commit()
                        
                        # Send completion update
                        send_project_agent_update(
                            project_id=project_uuid,
                            agent_id=agent.id,
                            message={
                                "type": "agent_status",
                                "data": {
                                    "project_id": project_uuid,
                                    "agent_id": agent.id,
                                    "status": "completed",
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                        )
                    else:
                        # Handle error
                        agent_result = AgentResultModel(
                            agent_id=agent.id,
                            project_id=agent.project_id,
                            status="error",
                            results=storable_data,
                        )
                        session.add(agent_result)
                        
                        # Update agent status
                        agent.agent_status = "error"
                        session.commit()
                        
                        # Send error update
                        send_project_agent_update(
                            project_id=project_uuid,
                            agent_id=agent.id,
                            message={
                                "type": "agent_status",
                                "data": {
                                    "project_id": project_uuid,
                                    "agent_id": agent.id,
                                    "status": "error",
                                    "error": storable_data["error"],
                                    "timestamp": datetime.utcnow().isoformat()
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
            elif agent.agent_platform == "twitter":
                # Create new event loop for this task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Send processing status
                    send_project_agent_update(
                        project_id=project_uuid,
                        agent_id=agent.id,
                        message={
                            "type": "agent_status",
                            "data": {
                                "project_id": project_uuid,
                                "agent_id": agent.id,
                                "status": "processing",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    )

                    # Create async session for the workflow
                    async def run_workflow():
                        async with async_session() as async_db:
                            # Prepare initial state
                            initial_state = {
                                "agent_name": agent.agent_name,
                                "goals": agent.goals.split(",") if agent.goals else [],
                                "instructions": agent.instructions,
                                "description": agent.project.description or "",
                                "expectation": agent.expectations or "",
                                "target_audience": agent.project.target_audience or "",
                                "company_keywords": agent.project.keywords or [],
                                "min_likes": agent.advanced_settings.get("min_likes", 0),
                                "max_age_days": agent.advanced_settings.get("max_age_days", 7),
                                "hashtags": [],
                                "tweets": [],
                                "retries": 0,
                                "error": None,
                                "db": async_db
                            }
                            
                            result = await twitter_graph.ainvoke(initial_state)
                            return result
                    
                    # Run the workflow
                    result = loop.run_until_complete(run_workflow())
                    logger.info("Agent execution completed", agent_id=agent_id, result=result)
                    
                    # Extract storable data from result
                    storable_data = extract_storable_data(result)
                    
                    if not storable_data.get("error"):
                        # Store Twitter posts
                        for tweet in result.get("tweets", []):
                            twitter_post = TwitterPostModel(
                                agent_name=agent.agent_name,
                                goals=agent.goals.split(",") if agent.goals else [],
                                instructions=agent.instructions,
                                tweet_id=tweet["tweet_id"],
                                text=tweet["text"],
                                created_at=tweet["created_at"],
                                user_name=tweet["username"],
                                user_screen_name=tweet.get("user_screen_name", ""),
                                retweet_count=tweet["retweets"],
                                favorite_count=tweet["likes"],
                                relevance_score=tweet.get("relevance_score", 0.0),
                                hashtags=tweet.get("hashtags", [])
                            )
                            session.add(twitter_post)
                        
                        # Save agent results
                        agent_result = AgentResultModel(
                            agent_id=agent.id,
                            project_id=agent.project_id,
                            status="completed",
                            results=storable_data,
                        )
                        session.add(agent_result)
                        
                        # Update agent status
                        agent.agent_status = "completed"
                        session.commit()
                        
                        # Send completion update
                        send_project_agent_update(
                            project_id=project_uuid,
                            agent_id=agent.id,
                            message={
                                "type": "agent_status",
                                "data": {
                                    "project_id": project_uuid,
                                    "agent_id": agent.id,
                                    "status": "completed",
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                        )
                    else:
                        # Handle error
                        agent_result = AgentResultModel(
                            agent_id=agent.id,
                            project_id=agent.project_id,
                            status="error",
                            results=storable_data,
                        )
                        session.add(agent_result)
                        
                        # Update agent status
                        agent.agent_status = "error"
                        session.commit()
                        
                        # Send error update
                        send_project_agent_update(
                            project_id=project_uuid,
                            agent_id=agent.id,
                            message={
                                "type": "agent_status",
                                "data": {
                                    "project_id": project_uuid,
                                    "agent_id": agent.id,
                                    "status": "error",
                                    "error": storable_data["error"],
                                    "timestamp": datetime.utcnow().isoformat()
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
        logger.error("Agent processing failed", 
                    agent_id=agent_id, 
                    error=str(e), 
                    exc_info=True)
        
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
                        results={"error": str(e)},
                    )
                    session.add(agent_result)
                    session.commit()
                    
                    # Send error update
                    project_uuid = str(agent.project.uuid)
                    send_project_agent_update(
                        project_id=project_uuid,
                        agent_id=agent.id,
                        message={
                            "type": "agent_status",
                            "data": {
                                "project_id": project_uuid,
                                "agent_id": agent.id,
                                "error": str(e),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    )
        except Exception as inner_e:
            logger.error("Failed to handle error state", agent_id=agent_id, error=str(inner_e)) 

# Add a test task to verify SSE functionality
# @shared_task
# def test_sse_message(project_id: str, message: str = "Test message"):
#     """Test task to send SSE messages"""
#     try:
#         logger.info("Sending test SSE message", 
#                    project_id=project_id, 
#                    message=message)
        
#         # Send message through test-event endpoint without token
#         response = requests.post(
#             f"http://localhost:8000/sse/projects/{project_id}/test-event"
#         )
        
#         if response.status_code == 200:
#             logger.info("Successfully sent test message", 
#                        project_id=project_id)
#             return {"status": "success", "message": "Test message sent"}
#         else:
#             logger.error("Failed to send test message", 
#                         status_code=response.status_code,
#                         response=response.text)
#             return {"status": "error", "message": f"HTTP {response.status_code}"}
            
#     except Exception as e:
#         logger.error("Failed to send test message", 
#                     error=str(e), 
#                     exc_info=True)
#         return {"status": "error", "message": str(e)}

@shared_task
def test_async_sse_message(project_id: str):
    """Simple test task to send SSE message"""
    try:
        logger.info("Sending test message", 
                   project_id=project_id)
        
        # Create a simple test message
        # message = {
        #     "type": "test_event",
        #     "data": {
        #         "project_id": project_id,
        #         "message": "Test message from asyncio",
        #         "timestamp": datetime.utcnow().isoformat()
        #     }
        # }

        message={
            "type": "agent_status",
            "data": {
                "project_id": project_id,
                "agent_id": 152,
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        
        # Publish message to Redis channel
        redis_client.publish(
            f"sse:project:{project_id}",
            json.dumps(message)
        )
        
        logger.info("Successfully sent test message", 
                   project_id=project_id)
        return {"status": "success", "message": "Test message sent"}
            
    except Exception as e:
        logger.error("Failed to send test message", 
                    error=str(e), 
                    exc_info=True)
        return {"status": "error", "message": str(e)} 