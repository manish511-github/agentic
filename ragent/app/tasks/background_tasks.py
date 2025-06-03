from datetime import datetime, timedelta
import asyncio
from typing import Dict, Optional, List
import structlog
from app.models import AgentModel, ProjectModel
from app.rdagent import reddit_graph, AgentState
from app.websocket.manager import manager
from sqlalchemy import select

logger = structlog.get_logger()

class BackgroundTaskManager:
    def __init__(self):
        self.running_tasks: Dict[int, asyncio.Task] = {}
        self.agent_states: Dict[int, Dict] = {}
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent agent executions
        self.task_queue = asyncio.Queue()  # Queue for managing agent tasks
        self.is_processing = False
    
    async def _update_agent_state(self, agent: AgentModel, state: dict):
        """Update agent state and broadcast to connected clients"""
        self.agent_states[agent.id] = state
        
        # Broadcast update to all user's connections
        await manager.broadcast_agent_status(
            user_id=agent.owner_id,
            agent_id=agent.id,
            status=state
        )
    
    async def start_agent_task(self, agent: AgentModel, db):

        """Start background task for an agent"""
        if agent.id in self.running_tasks:
            logger.warning("Agent task already running", agent_id=agent.id)
            return
        
        # Add agent to queue
        await self.task_queue.put((agent, db))
        
        # Start processing if not already running
        if not self.is_processing:
            self.is_processing = True
            asyncio.create_task(self._process_queue())
        
        logger.info("Agent task queued", agent_id=agent.id)
    
    async def _process_queue(self):
        """Process the agent task queue"""
        try:
            while True:
                # Get next agent from queue
                agent, db = await self.task_queue.get()
                
                try:
                    # Create and start task with semaphore
                    async with self.semaphore:
                        task = asyncio.create_task(
                            self._run_agent_cycle(agent, db)
                        )
                        self.running_tasks[agent.id] = task
                        logger.info("Started agent task", agent_id=agent.id)
                        
                        # Wait for task to complete
                        await task
                        
                except Exception as e:
                    logger.error("Error processing agent", agent_id=agent.id, error=str(e))
                finally:
                    self.task_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info("Queue processing cancelled")
        finally:
            self.is_processing = False
    
    async def stop_agent_task(self, agent_id: int):
        """Stop background task for an agent"""
        if agent_id in self.running_tasks:
            self.running_tasks[agent_id].cancel()
            del self.running_tasks[agent_id]
            logger.info("Stopped background task", agent_id=agent_id)
    
    async def stop_all_tasks(self):
        """Stop all running agent tasks"""
        for agent_id in list(self.running_tasks.keys()):
            await self.stop_agent_task(agent_id)
        logger.info("All agent tasks stopped")
    
    async def get_running_agents(self) -> List[Dict]:
        """Get information about all running agents"""
        return [
            {
                'agent_id': agent_id,
                'status': self.agent_states.get(agent_id, {}).get('status', 'unknown'),
                'last_run': self.agent_states.get(agent_id, {}).get('last_run'),
                'posts_fetched': self.agent_states.get(agent_id, {}).get('posts_fetched', 0)
            }
            for agent_id in self.running_tasks.keys()
        ]
    
    async def _run_agent_cycle(self, agent: AgentModel, db):
        """Run the agent's task cycle"""
        try:
            # Run immediately on first execution
            await self._execute_agent(agent, db)
            
            while True:
                # Calculate next run time
                next_run = self._calculate_next_run(agent)
                wait_time = (next_run - datetime.now()).total_seconds()
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Run the agent
                await self._execute_agent(agent, db)
                
        except asyncio.CancelledError:
            logger.info("Agent task cancelled", agent_id=agent.id)
        except Exception as e:
            logger.error("Agent task failed", agent_id=agent.id, error=str(e))
    
    def _calculate_next_run(self, agent: AgentModel) -> datetime:
        """Calculate next run time based on agent settings"""
        now = datetime.now()
        if agent.review_period == "daily":
            return now + timedelta(days=1)
        elif agent.review_period == "weekly":
            return now + timedelta(weeks=1)
        elif agent.review_period == "monthly":
            return now + timedelta(days=30)
        return now + timedelta(days=1)  # Default to daily
    
    async def _execute_agent(self, agent: AgentModel, db):
        """Execute the Reddit agent"""
        try:
            # Fetch project details
            result = await db.execute(
                select(ProjectModel).filter(ProjectModel.id == agent.project_id)
            )
            project = result.scalars().first()
            
            if not project:
                raise ValueError(f"Project with id {agent.project_id} not found")
            
            # Create initial state with project details
            initial_state = AgentState(
                agent_name=agent.agent_name,
                goals=agent.goals,
                instructions=agent.instructions,
                description=project.description,  # From project
                expectation=agent.expectations,
                target_audience=project.target_audience,  # From project
                company_keywords=project.keywords,  # From project
                min_upvotes=agent.min_upvotes,
                max_age_days=agent.max_age_days,
                restrict_to_goal_subreddits=agent.restrict_to_goal_subreddits,
                subreddits=[],
                posts=[],
                retries=0,
                error=None,
                db=db
            )

            
            # Update status to running
            await self._update_agent_state(agent, {
                'status': 'running',
                'last_run': datetime.now(),
                'error': None
            })
            
            # Run the agent
            result = await reddit_graph.ainvoke(initial_state)
            
            # Update state with success
            await self._update_agent_state(agent, {
                'last_run': datetime.now(),
                'status': 'success',
                'posts_fetched': len(result.get('posts', [])),
                'error': result.get('error')
            })
            
            # Broadcast new posts if any
            if result.get('posts'):
                await manager.broadcast_agent_update(
                    user_id=agent.owner_id,
                    agent_id=agent.id,
                    data={'posts': result.get('posts')}
                )
            
            logger.info(
                "Agent execution completed",
                agent_id=agent.id,
                project_id=agent.project_id,
                posts_fetched=len(result.get('posts', [])),
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error("Agent execution failed", agent_id=agent.id, error=str(e))
            await self._update_agent_state(agent, {
                'last_run': datetime.now(),
                'status': 'error',
                'error': str(e)
            })

# Create and export the task manager instance
task_manager = BackgroundTaskManager() 