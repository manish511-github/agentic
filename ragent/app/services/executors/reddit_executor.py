import asyncio
from typing import Dict, Any
import structlog

from .base_executor import BaseAgentExecutor
from .result_mappers import get_result_mapper
from app.core.agents_tasks.rdagent.rd_agent_v1.graph import reddit_graph

logger = structlog.get_logger()


class RedditAgentExecutor(BaseAgentExecutor):
    """
    Executor for Reddit agents.

    Handles the execution of Reddit marketing agents by:
    - Creating appropriate initial state from agent and project data
    - Running the Reddit agent workflow
    - Saving results using the Reddit result mapper
    """

    def create_initial_state(self) -> Dict[str, Any]:
        """
        Create the initial state for the Reddit agent workflow.

        Returns:
            Dict[str, Any]: Initial state dictionary with all required fields
        """
        # Get project data if available
        project = self.agent.project if hasattr(
            self.agent, 'project') and self.agent.project else None

        # Create initial state with all required fields for Reddit agent
        initial_state = {
            "agent_name": self.agent.agent_name,
            "goals": self.agent.goals.split(",") if self.agent.goals else [],
            "instructions": self.agent.instructions or "",
            "description": project.description if project else "",
            "expectation": self.agent.expectations or "",
            "target_audience": project.target_audience if project else "",
            "company_keywords": project.keywords if project else [],
            "min_upvotes": self.agent.advanced_settings.get("min_upvotes", 0) if self.agent.advanced_settings else 0,
            "max_age_days": self.agent.advanced_settings.get("max_age_days", 7) if self.agent.advanced_settings else 7,
            "restrict_to_goal_subreddits": self.agent.advanced_settings.get("restrict_to_goal_subreddits", False) if self.agent.advanced_settings else False,
            "subreddits": [],
            "posts": [],
            "retries": 0,
            "error": None,
        }

        logger.info(
            "Created Reddit agent initial state",
            execution_id=self.execution_id,
            agent_id=self.agent_id,
            agent_name=self.agent.agent_name,
            goals=initial_state["goals"],
            keywords_count=len(initial_state["company_keywords"])
        )

        return initial_state

    async def execute_agent(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Reddit agent workflow.

        Args:
            initial_state: Initial state for the Reddit workflow

        Returns:
            Dict[str, Any]: Workflow execution results
        """
        try:
            logger.info(
                "Starting Reddit agent workflow execution",
                execution_id=self.execution_id,
                agent_id=self.agent_id
            )

            # Execute the Reddit agent workflow
            result = await reddit_graph.ainvoke(initial_state)

            # Validate results
            if not isinstance(result, dict):
                raise ValueError("Reddit agent returned invalid result format")

            logger.info(
                "Reddit agent workflow completed successfully",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                posts_found=len(result.get("posts", [])),
                subreddits_processed=len(result.get("subreddits", []))
            )

            return result

        except Exception as e:
            error_msg = f"Reddit agent workflow failed: {str(e)}"
            logger.error(
                "Reddit agent workflow execution failed",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            return {"error": error_msg}

    def save_results(self, results: Dict[str, Any]) -> bool:
        """
        Save Reddit agent results using the Reddit result mapper.

        Args:
            results: Results from Reddit agent execution

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Get the Reddit result mapper
            result_mapper = get_result_mapper(
                platform="reddit",
                db_session=self.db_session,
                execution_id=self.execution_id,
                agent=self.agent
            )

            # Save results
            success = result_mapper.save_results(results)

            if success:
                logger.info(
                    "Reddit agent results saved successfully",
                    execution_id=self.execution_id,
                    agent_id=self.agent_id,
                    posts_saved=len(results.get("posts", []))
                )
            else:
                logger.error(
                    "Failed to save Reddit agent results",
                    execution_id=self.execution_id,
                    agent_id=self.agent_id
                )

            return success

        except Exception as e:
            logger.error(
                "Error saving Reddit agent results",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            return False
