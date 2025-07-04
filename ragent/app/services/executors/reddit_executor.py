import asyncio
from typing import Dict, Any
import structlog

from .base_executor import BaseAgentExecutor
from .result_mappers import get_result_mapper
from app.core.agents_tasks.rdagent.rd_agent_v1.graph import parallel_reddit_graph
from app.core.agents_tasks.rdagent.rd_agent_v1.state import RedditAgentState
from app.core.llm_client import get_llm
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class RedditAgentExecutor(BaseAgentExecutor):
    """
    Executor for Reddit agents.

    Handles the execution of Reddit marketing agents by:
    - Creating appropriate initial state from agent and project data
    - Running the Reddit agent workflow
    - Saving results using the Reddit result mapper
    """

    def create_initial_state(self) -> RedditAgentState:
        """
        Create the initial state for the Reddit agent workflow.

        Returns:
            RedditAgentState: Initial state dictionary with all required fields
        """
        # Get project data if available
        project = self.agent.project if hasattr(
            self.agent, 'project') and self.agent.project else None

        # Handle company_keywords - convert from JSON to List[str]
        company_keywords = []
        if project and project.keywords:
            if isinstance(project.keywords, list):
                company_keywords = project.keywords
            elif isinstance(project.keywords, str):
                # If it's a string, try to parse as JSON or split by comma
                try:
                    import json
                    company_keywords = json.loads(project.keywords)
                except (json.JSONDecodeError, TypeError):
                    company_keywords = [kw.strip() for kw in project.keywords.split(",") if kw.strip()]
            else:
                company_keywords = []

        # Handle agent keywords - ensure it's a list
        agent_keywords = []
        if self.agent.agent_keywords:
            if isinstance(self.agent.agent_keywords, list):
                agent_keywords = self.agent.agent_keywords
            else:
                agent_keywords = []

        # Convert database session to AsyncSession for compatibility
        # Note: The base executor uses sync Session, but AgentState expects AsyncSession
        # We'll pass None for now and let the workflow handle db access
        async_db_session = None  # Will be handled by the workflow nodes

        # Get LLM instance
        llm = get_llm()

        # Create initial state with all required fields for Reddit agent
        initial_state = RedditAgentState(
            agent_name=self.agent.agent_name,
            agent_id=self.agent.id,
            agent_platform=self.agent.agent_platform,
            project_id=self.agent.project_id,
            execution_id=self.execution_id,
            goals=self.agent.goals.split(",") if self.agent.goals else [],
            description=project.description if project else "",
            expectation=self.agent.expectations or "",
            target_audience=project.target_audience if project else "",
            instructions=self.agent.instructions or "",
            company_keywords=company_keywords,
            keywords=agent_keywords,
            generated_queries=None,
            min_upvotes=self.agent.platform_settings.get("reddit", {}).get("minUpvotes", 0) if self.agent.platform_settings else 0,
            max_age_days=self.agent.platform_settings.get("reddit", {}).get("timeRange", "day") if self.agent.platform_settings else 7,
            restrict_to_goal_subreddits=self.agent.platform_settings.get("reddit", {}).get("monitorComments", False) if self.agent.platform_settings else False,
            subreddits=self.agent.platform_settings.get("reddit", {}).get("subreddits", []),
            posts=[],
            seen_post_ids=set(),
            direct_posts=[],
            subreddit_posts=[],
            retries=0,
            error=None,
            db=async_db_session,
            llm=llm,
        )

        logger.info(
            "Created Reddit agent initial state",
            execution_id=self.execution_id,
            agent_id=self.agent_id,
            agent_name=self.agent.agent_name,
            goals=initial_state["goals"],
            keywords_count=len(initial_state["company_keywords"]),
            agent_keywords_count=len(initial_state["keywords"]) if initial_state["keywords"] else 0
        )

        return initial_state

    async def execute_agent(self, initial_state: RedditAgentState) -> Any:
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
            result = await parallel_reddit_graph.ainvoke(initial_state)

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
