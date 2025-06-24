import asyncio
from typing import Dict, Any
import structlog

from .base_executor import BaseAgentExecutor
from .result_mappers import get_result_mapper
from app.core.agents_tasks.hn_agent.hnagent import hn_workflow
from app.core.agents_tasks.hn_agent.models import HNAgentInput

logger = structlog.get_logger()


class HackerNewsAgentExecutor(BaseAgentExecutor):
    """
    Executor for Hacker News agents.

    Handles the execution of Hacker News marketing agents by:
    - Creating appropriate initial state from agent and project data
    - Running the Hacker News agent workflow
    - Saving results using the Hacker News result mapper
    """

    def create_initial_state(self) -> Dict[str, Any]:
        """
        Create the initial state for the Hacker News agent workflow.

        Returns:
            Dict[str, Any]: Initial state dictionary with all required fields for HN agent
        """
        # Get project data if available
        project = self.agent.project if hasattr(
            self.agent, 'project') and self.agent.project else None

        # Create HN agent input model
        hn_agent_input = HNAgentInput(
            agent_name=self.agent.agent_name,
            goals=self.agent.goals.split(",") if self.agent.goals else [],
            instructions=self.agent.instructions or "",
            company_keywords=project.keywords if project else [],
            description=project.description if project else "",
            expectation=self.agent.expectations or "",
            target_audience=project.target_audience if project else "",
            min_score=self.agent.advanced_settings.get(
                "min_score", 10) if self.agent.advanced_settings else 10,
            max_age_days=self.agent.advanced_settings.get(
                "max_age_days", 7) if self.agent.advanced_settings else 7
        )

        # Create initial state for HN workflow
        initial_state = {
            "agent_input": hn_agent_input,
            "expanded_queries": [],
            "stories_stored": 0,
            "search_results": [],
            "filtered_stories": [],
            "summarized_stories": [],
            "final_stories_output": None
        }

        logger.info(
            "Created Hacker News agent initial state",
            execution_id=self.execution_id,
            agent_id=self.agent_id,
            agent_name=self.agent.agent_name,
            goals=hn_agent_input.goals,
            keywords_count=len(hn_agent_input.company_keywords),
            min_score=hn_agent_input.min_score,
            max_age_days=hn_agent_input.max_age_days
        )

        return initial_state

    async def execute_agent(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Hacker News agent workflow.

        Args:
            initial_state: Initial state for the HN workflow

        Returns:
            Dict[str, Any]: Workflow execution results
        """
        try:
            logger.info(
                "Starting Hacker News agent workflow execution",
                execution_id=self.execution_id,
                agent_id=self.agent_id
            )

            # Execute the Hacker News agent workflow
            result = await hn_workflow.ainvoke(initial_state)

            # Validate results
            if not isinstance(result, dict):
                raise ValueError(
                    "Hacker News agent returned invalid result format")

            stories = result.get("final_stories_output", [])
            logger.info(
                "Hacker News agent workflow completed successfully",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                stories_found=len(stories) if stories else 0
            )

            return result

        except Exception as e:
            error_msg = f"Hacker News agent workflow failed: {str(e)}"
            logger.error(
                "Hacker News agent workflow execution failed",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            return {"error": error_msg}

    def save_results(self, results: Dict[str, Any]) -> bool:
        """
        Save Hacker News agent results using the HN result mapper.

        Args:
            results: Results from Hacker News agent execution

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Get the Hacker News result mapper
            result_mapper = get_result_mapper(
                platform="hackernews",
                db_session=self.db_session,
                execution_id=self.execution_id,
                agent=self.agent
            )

            # Save results
            success = result_mapper.save_results(results)

            if success:
                stories = results.get("final_stories_output", [])
                logger.info(
                    "Hacker News agent results saved successfully",
                    execution_id=self.execution_id,
                    agent_id=self.agent_id,
                    stories_saved=len(stories) if stories else 0
                )
            else:
                logger.error(
                    "Failed to save Hacker News agent results",
                    execution_id=self.execution_id,
                    agent_id=self.agent_id
                )

            return success

        except Exception as e:
            logger.error(
                "Error saving Hacker News agent results",
                execution_id=self.execution_id,
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            return False
