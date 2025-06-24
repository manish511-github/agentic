from abc import ABC, abstractmethod
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime
import structlog
from sqlalchemy import select

from app.models import (
    AgentResultModel,
    RedditPostModel,
    RedditAgentExecutionMapperModel,
    AgentModel,
    ExecutionModel,
    ExecutionStatusEnum
)

logger = structlog.get_logger()


class ResultMapper(ABC):
    """
    Abstract base class for result mappers.

    Result mappers are responsible for saving agent execution results
    to the appropriate database tables in a structured way.
    """

    def __init__(self, db_session: Session, execution_id: int, agent: AgentModel):
        """
        Initialize the result mapper.

        Args:
            db_session: Database session for operations
            execution_id: The execution ID associated with these results
            agent: The agent that generated these results
        """
        self.db_session = db_session
        self.execution_id = execution_id
        self.agent = agent

    @abstractmethod
    def save_results(self, results: Dict[str, Any]) -> bool:
        """
        Save the execution results to the database.

        Args:
            results: Results dictionary from agent execution

        Returns:
            bool: True if save successful, False otherwise
        """
        pass

    def save_agent_result(self, results: Dict[str, Any]) -> bool:
        """
        Save general agent results to the agent_results table.

        Args:
            results: Results dictionary to save

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Extract storable data (remove non-serializable items)
            storable_data = self._extract_storable_data(results)

            # change this to save the agent result in the execution table
            execution = self.db_session.execute(select(ExecutionModel).where(
                ExecutionModel.id == self.execution_id)).scalar_one_or_none()
            if execution is None:
                logger.error(
                    f"Execution with id {self.execution_id} not found")
                return False
            execution.results = storable_data
            execution.status = ExecutionStatusEnum.completed

            self.db_session.add(execution)

            logger.info(
                "Agent result saved successfully",
                execution_id=self.execution_id,
                agent_id=self.agent.id
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to save agent result",
                execution_id=self.execution_id,
                agent_id=self.agent.id,
                error=str(e)
            )
            return False

    def _extract_storable_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only serializable data from results.

        Args:
            results: Raw results dictionary

        Returns:
            Dict[str, Any]: Clean, serializable data
        """
        if results.get("error"):
            return {"error": str(results["error"])}

        # Remove non-serializable keys like 'db', 'llm', etc.
        non_serializable_keys = {'db', 'llm', 'session'}

        clean_data = {
            k: v for k, v in results.items()
            if k not in non_serializable_keys and v is not None
        }

        return clean_data


class RedditResultMapper(ResultMapper):
    """
    Result mapper for Reddit agent executions.

    Handles saving Reddit-specific results including posts and execution mappings.
    """

    def save_results(self, results: Dict[str, Any]) -> bool:
        """
        Save Reddit agent results to the database.

        Args:
            results: Results from Reddit agent execution

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Save general agent results
            if not self.save_agent_result(results):
                return False

            # Save Reddit-specific data
            posts = results.get("posts", [])
            if posts:
                self._save_reddit_posts(posts)
                self._save_reddit_execution_mappings(posts)

            self.db_session.commit()

            logger.info(
                "Reddit results saved successfully",
                execution_id=self.execution_id,
                agent_id=self.agent.id,
                posts_count=len(posts)
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to save Reddit results",
                execution_id=self.execution_id,
                agent_id=self.agent.id,
                error=str(e)
            )
            self.db_session.rollback()
            return False

    def _save_reddit_posts(self, posts: List[Dict[str, Any]]):
        """
        Save Reddit posts to the reddit_posts table.

        Args:
            posts: List of post dictionaries
        """
        for post in posts:
            try:
                # Check if post already exists
                existing_post = self.db_session.query(RedditPostModel).filter(
                    RedditPostModel.post_id == post["post_id"]
                ).first()

                if existing_post:
                    logger.debug(
                        "Reddit post already exists, skipping",
                        post_id=post["post_id"]
                    )
                    continue

                reddit_post = RedditPostModel(
                    agent_name=self.agent.agent_name,
                    goals=self.agent.goals.split(
                        ",") if self.agent.goals else [],
                    instructions=self.agent.instructions,
                    subreddit=post["subreddit"],
                    post_id=post["post_id"],
                    post_title=post["post_title"],
                    post_body=post["post_body"],
                    post_url=post["post_url"],
                    upvotes=post.get("upvotes", 0),
                    comment_count=post.get("comment_count", 0),
                    created=datetime.fromisoformat(
                        post["created"]) if post.get("created") else None,
                    keyword_relevance=post.get("keyword_relevance"),
                    matched_query=post.get("matched_query"),
                    semantic_relevance=post.get("semantic_relevance"),
                    combined_relevance=post.get("combined_relevance")
                )

                self.db_session.add(reddit_post)

            except Exception as e:
                logger.warning(
                    "Failed to save Reddit post",
                    post_id=post.get("post_id", "unknown"),
                    error=str(e)
                )

    def _save_reddit_execution_mappings(self, posts: List[Dict[str, Any]]):
        """
        Save Reddit execution mappings to track which posts were found in which execution.

        Args:
            posts: List of post dictionaries
        """
        for post in posts:
            try:
                # Check if mapping already exists
                existing_mapping = self.db_session.query(RedditAgentExecutionMapperModel).filter(
                    RedditAgentExecutionMapperModel.execution_id == self.execution_id,
                    RedditAgentExecutionMapperModel.agent_id == self.agent.id,
                    RedditAgentExecutionMapperModel.post_id == post["post_id"]
                ).first()

                if existing_mapping:
                    logger.debug(
                        "Reddit execution mapping already exists, skipping",
                        execution_id=self.execution_id,
                        post_id=post["post_id"]
                    )
                    continue

                mapping = RedditAgentExecutionMapperModel(
                    execution_id=self.execution_id,
                    agent_id=self.agent.id,
                    post_id=post["post_id"],
                    relevance_score=post.get("combined_relevance", 0.0),
                    comment_draft=post.get("comment_draft"),
                    status="processed"
                )

                self.db_session.add(mapping)

            except Exception as e:
                logger.warning(
                    "Failed to save Reddit execution mapping",
                    execution_id=self.execution_id,
                    post_id=post.get("post_id", "unknown"),
                    error=str(e)
                )


class HackerNewsResultMapper(ResultMapper):
    """
    Result mapper for Hacker News agent executions.

    Currently saves only general agent results.
    Future enhancement: Add HN-specific tables and mappings.
    """

    def save_results(self, results: Dict[str, Any]) -> bool:
        """
        Save Hacker News agent results to the database.

        Args:
            results: Results from Hacker News agent execution

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # For now, just save general agent results
            # TODO: Add HN-specific result saving when tables are available
            if not self.save_agent_result(results):
                return False

            self.db_session.commit()

            stories = results.get("final_stories_output", [])
            logger.info(
                "Hacker News results saved successfully",
                execution_id=self.execution_id,
                agent_id=self.agent.id,
                stories_count=len(stories) if stories else 0
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to save Hacker News results",
                execution_id=self.execution_id,
                agent_id=self.agent.id,
                error=str(e)
            )
            self.db_session.rollback()
            return False


def get_result_mapper(platform: str, db_session: Session, execution_id: int, agent: AgentModel) -> ResultMapper:
    """
    Factory function to get the appropriate result mapper for a platform.

    Args:
        platform: Agent platform (e.g., 'reddit', 'hackernews')
        db_session: Database session
        execution_id: Execution ID
        agent: Agent model instance

    Returns:
        ResultMapper: Appropriate result mapper instance

    Raises:
        ValueError: If platform is not supported
    """
    mappers = {
        "reddit": RedditResultMapper,
        "hackernews": HackerNewsResultMapper,
    }

    mapper_class = mappers.get(platform.lower())
    if not mapper_class:
        raise ValueError(f"Unsupported platform: {platform}")

    return mapper_class(db_session, execution_id, agent)
