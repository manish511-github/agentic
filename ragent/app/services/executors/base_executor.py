from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
import structlog
from datetime import datetime

from app.models import AgentModel, ExecutionModel, ExecutionStatusEnum
from app.database import SessionLocal

logger = structlog.get_logger()


class BaseAgentExecutor(ABC):
    """
    Abstract base class for agent executors.

    Provides common functionality for all agent executors including:
    - Database session management
    - Execution state management
    - Error handling
    - Logging
    """

    def __init__(self, execution_id: int, agent_id: int):
        """
        Initialize the executor with execution and agent IDs.

        Args:
            execution_id: The execution ID to process
            agent_id: The agent ID to execute
        """
        self.execution_id = execution_id
        self.agent_id = agent_id
        self.db_session: Optional[Session] = None
        self.execution: Optional[ExecutionModel] = None
        self.agent: Optional[AgentModel] = None

    def __enter__(self):
        """Context manager entry - initialize database session."""
        self.db_session = SessionLocal()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup database session."""
        if self.db_session:
            if exc_type is not None:
                # Rollback on error
                self.db_session.rollback()
                logger.error(
                    "Executor failed, rolling back transaction",
                    execution_id=self.execution_id,
                    agent_id=self.agent_id,
                    error=str(exc_val)
                )
            self.db_session.close()

    def initialize(self) -> bool:
        """
        Initialize the executor by loading execution and agent data.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.db_session:
            logger.error("Database session not initialized")
            return False

        try:
            # Load execution with agent relationship
            self.execution = self.db_session.query(ExecutionModel).filter(
                ExecutionModel.id == self.execution_id
            ).first()

            if not self.execution:
                logger.error(
                    "Execution not found",
                    execution_id=self.execution_id
                )
                return False

            # Load agent with project relationship
            self.agent = self.db_session.query(AgentModel).filter(
                AgentModel.id == self.agent_id
            ).first()

            if not self.agent:
                logger.error(
                    "Agent not found",
                    agent_id=self.agent_id
                )
                return False

            # Mark execution as running
            self.execution.status = ExecutionStatusEnum.running
            self.db_session.commit()

            logger.info(
                "Executor initialized successfully",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                agent_platform=self.agent.agent_platform
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to initialize executor",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                error=str(e)
            )
            return False

    def mark_execution_completed(self, results: Optional[Dict[str, Any]] = None):
        """
        Mark the execution as completed and save results.

        Args:
            results: Optional results dictionary to store
        """
        if self.execution and self.db_session:
            self.execution.status = ExecutionStatusEnum.completed
            if results:
                self.execution.results = results
            self.db_session.commit()

            logger.info(
                "Execution marked as completed",
                execution_id=self.execution_id,
                agent_id=self.agent_id
            )

    def mark_execution_failed(self, error_message: str):
        """
        Mark the execution as failed and save error message.

        Args:
            error_message: Error message to store
        """
        if self.execution and self.db_session:
            self.execution.status = ExecutionStatusEnum.failed
            self.execution.results = {"error": error_message}
            self.db_session.commit()

            logger.error(
                "Execution marked as failed",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                error=error_message
            )

    @abstractmethod
    def create_initial_state(self) -> Dict[str, Any]:
        """
        Create the initial state for the agent workflow.

        Returns:
            Dict[str, Any]: Initial state dictionary
        """
        pass

    @abstractmethod
    async def execute_agent(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent workflow.

        Args:
            initial_state: Initial state for the workflow

        Returns:
            Dict[str, Any]: Workflow results
        """
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any]) -> bool:
        """
        Save the execution results to the database.

        Args:
            results: Results from the agent execution

        Returns:
            bool: True if save successful, False otherwise
        """
        pass

    async def run(self) -> Dict[str, Any]:
        """
        Main execution method that orchestrates the entire agent execution process.

        Returns:
            Dict[str, Any]: Execution results
        """
        try:
            # Initialize
            if not self.initialize():
                return {"error": "Failed to initialize executor"}

            # Create initial state
            initial_state = self.create_initial_state()

            logger.info(
                "Starting agent execution",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                agent_platform=self.agent.agent_platform
            )

            # Execute agent
            results = await self.execute_agent(initial_state)

            # Check for errors in results
            if results.get("error"):
                self.mark_execution_failed(results["error"])
                return results

            # Save results
            if self.save_results(results):
                self.mark_execution_completed(results)
                return results
            else:
                error_msg = "Failed to save execution results"
                self.mark_execution_failed(error_msg)
                return {"error": error_msg}

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(
                "Agent execution failed",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            self.mark_execution_failed(error_msg)
            return {"error": error_msg}
