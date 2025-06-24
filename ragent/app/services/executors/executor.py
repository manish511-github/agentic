import asyncio
import nest_asyncio
from typing import Dict, Any, Optional
import structlog

from app.celery_app import celery_app
from .executor_factory import ExecutorFactory
from app.models import AgentModel, ExecutionModel, ExecutionStatusEnum
from app.database import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import joinedload

# Apply nest_asyncio to allow nested event loops in Celery
nest_asyncio.apply()

logger = structlog.get_logger()


def _get_execution_and_agent(execution_id: int, agent_id: int) -> tuple[Optional[ExecutionModel], Optional[AgentModel]]:
    """
    Retrieve execution and agent data from database.

    Args:
        execution_id: Execution ID to retrieve
        agent_id: Agent ID to retrieve

    Returns:
        tuple: (ExecutionModel, AgentModel) or (None, None) if not found
    """
    with SessionLocal() as db_session:
        try:
            # Get execution with relationships
            execution = db_session.execute(
                select(ExecutionModel)
                .where(ExecutionModel.id == execution_id)
                .options(joinedload(ExecutionModel.agent))
            ).scalar_one_or_none()

            if not execution:
                logger.error(
                    "Execution not found",
                    execution_id=execution_id
                )
                return None, None

            # Get agent with relationships
            agent = db_session.execute(
                select(AgentModel)
                .where(AgentModel.id == agent_id)
                .options(joinedload(AgentModel.project))
            ).scalar_one_or_none()

            if not agent:
                logger.error(
                    "Agent not found",
                    agent_id=agent_id
                )
                return None, None

            logger.info(
                "Successfully retrieved execution and agent data",
                execution_id=execution_id,
                agent_id=agent_id,
                agent_platform=agent.agent_platform
            )

            return execution, agent

        except Exception as e:
            logger.error(
                "Failed to retrieve execution and agent data",
                execution_id=execution_id,
                agent_id=agent_id,
                error=str(e)
            )
            return None, None


async def _execute_agent_async(execution_id: int, agent_id: int, platform: str) -> Dict[str, Any]:
    """
    Execute agent asynchronously using the appropriate executor.

    Args:
        execution_id: Execution ID to process
        agent_id: Agent ID to execute
        platform: Agent platform

    Returns:
        Dict[str, Any]: Execution results
    """
    try:
        # Create the appropriate executor using the factory
        executor = ExecutorFactory.create_executor(
            platform, execution_id, agent_id)

        # Execute the agent using context manager for proper cleanup
        with executor:
            result = await executor.run()
            return result

    except Exception as e:
        error_msg = f"Agent execution failed: {str(e)}"
        logger.error(
            "Agent execution failed",
            execution_id=execution_id,
            agent_id=agent_id,
            platform=platform,
            error=str(e),
            exc_info=True
        )
        return {"error": error_msg}


@celery_app.task(
    name="app.services.executors.executor.run_agent",
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def run_agent(self, execution_id: int, agent_id: int) -> Dict[str, Any]:
    """
    Main Celery task for executing agents.

    This task:
    1. Retrieves execution and agent details from the database
    2. Validates the agent platform and project configuration
    3. Creates and runs the appropriate agent executor
    4. Handles errors and retries appropriately

    Args:
        execution_id: The execution ID to process
        agent_id: The agent ID to execute

    Returns:
        Dict[str, Any]: Execution results
    """
    logger.info(
        "Starting agent execution task",
        execution_id=execution_id,
        agent_id=agent_id,
        task_id=self.request.id
    )

    try:
        # Step 1: Retrieve execution and agent data
        execution, agent = _get_execution_and_agent(execution_id, agent_id)

        if not execution or not agent:
            error_msg = "Failed to retrieve execution or agent data"
            logger.error(
                error_msg,
                execution_id=execution_id,
                agent_id=agent_id
            )
            return {"error": error_msg}

        # Step 2: Validate agent configuration
        if not agent.agent_platform:
            error_msg = f"Agent platform not specified for agent {agent_id}"
            logger.error(error_msg, agent_id=agent_id)
            return {"error": error_msg}

        if not agent.project:
            error_msg = f"Agent {agent_id} has no associated project"
            logger.error(error_msg, agent_id=agent_id)
            return {"error": error_msg}

        # Step 3: Check if platform is supported
        try:
            supported_platforms = ExecutorFactory.get_supported_platforms()
            if agent.agent_platform.lower() not in supported_platforms:
                error_msg = (
                    f"Unsupported agent platform: {agent.agent_platform}. "
                    f"Supported platforms: {', '.join(supported_platforms)}"
                )
                logger.error(error_msg, agent_id=agent_id,
                             platform=agent.agent_platform)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Failed to validate platform: {str(e)}"
            logger.error(error_msg, agent_id=agent_id, error=str(e))
            return {"error": error_msg}

        # Step 4: Execute the agent
        logger.info(
            "Executing agent",
            execution_id=execution_id,
            agent_id=agent_id,
            platform=agent.agent_platform,
            agent_name=agent.agent_name
        )

        # Create event loop for async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _execute_agent_async(
                    execution_id, agent_id, agent.agent_platform)
            )
        finally:
            loop.close()

        # Step 5: Handle results
        if result.get("error"):
            logger.error(
                "Agent execution completed with error",
                execution_id=execution_id,
                agent_id=agent_id,
                error=result["error"]
            )
        else:
            logger.info(
                "Agent execution completed successfully",
                execution_id=execution_id,
                agent_id=agent_id,
                platform=agent.agent_platform
            )

        return result

    except Exception as e:
        error_msg = f"Task execution failed: {str(e)}"
        logger.error(
            "Task execution failed",
            execution_id=execution_id,
            agent_id=agent_id,
            error=str(e),
            exc_info=True
        )

        # Retry the task if we haven't exceeded max retries
        if self.request.retries < self.max_retries:
            logger.info(
                "Retrying task",
                execution_id=execution_id,
                agent_id=agent_id,
                retry_count=self.request.retries + 1,
                max_retries=self.max_retries
            )
            raise self.retry(countdown=60)

        return {"error": error_msg}
