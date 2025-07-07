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
from app.core.agents_tasks.rdagent.schemas import RedditPost

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

        clean_data = {}
        for k, v in results.items():
            if k not in non_serializable_keys and v is not None:
                # Convert non-serializable types to serializable ones
                clean_data[k] = self._make_serializable(v)

        return clean_data

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to serializable ones.

        Args:
            obj: Object to make serializable

        Returns:
            Serializable version of the object
        """
        if isinstance(obj, set):
            # Convert sets to lists
            return list(obj)
        elif isinstance(obj, dict):
            # Recursively handle dictionaries
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively handle lists and tuples
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'dict') and callable(obj.dict):
            # Handle Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            # Handle other objects with __dict__
            try:
                return {k: self._make_serializable(v) for k, v in obj.__dict__.items() 
                       if not k.startswith('_')}
            except:
                # If conversion fails, convert to string
                return str(obj)
        else:
            # For basic types (str, int, float, bool, None) and unknown types
            try:
                # Test if it's JSON serializable
                import json
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # If not serializable, convert to string
                return str(obj)


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
            # Save general agent results first
            if not self.save_agent_result(results):
                return False

            # Extract posts from results (could be in multiple locations)
            posts = results.get("posts", [])
            if not posts:
                # Try alternative locations in case of different agent versions
                # Combine posts from multiple sources and deduplicate by post_id
                all_posts = results.get("subreddit_posts", []) + results.get("direct_posts", [])
                
                # Deduplicate posts by post_id (keep first occurrence)
                seen_post_ids = set()
                posts = []
                duplicates_found = 0
                
                for post in all_posts:
                    post_id = getattr(post, 'post_id', None)
                    if post_id and post_id not in seen_post_ids:
                        posts.append(post)
                        seen_post_ids.add(post_id)
                    elif post_id in seen_post_ids:
                        duplicates_found += 1
                        logger.debug(
                            "Duplicate post found during extraction, skipping",
                            post_id=post_id
                        )
                
                if duplicates_found > 0:
                    logger.info(
                        "Deduplicated posts during extraction",
                        total_from_sources=len(all_posts),
                        unique_posts=len(posts),
                        duplicates_removed=duplicates_found
                    )

            # Save Reddit-specific data if we have posts
            if posts:
                logger.info(f"Saving {len(posts)} Reddit posts and execution mappings")
                
                # Save posts first (they're referenced by execution mappings)
                self._save_reddit_posts(posts)
                
                # Flush to ensure posts are saved before mappings reference them
                self.db_session.flush()
                
                # Save execution mappings that reference the posts
                self._save_reddit_execution_mappings(posts)
                
                # Commit all changes
                self.db_session.commit()
                
                logger.info(
                    "Reddit results saved successfully",
                    execution_id=self.execution_id,
                    agent_id=self.agent.id,
                    posts_count=len(posts)
                )
            else:
                # No posts found, but still commit the general results
                self.db_session.commit()
                logger.info(
                    "Reddit agent completed with no posts found",
                    execution_id=self.execution_id,
                    agent_id=self.agent.id
                )

            return True

        except Exception as e:
            logger.error(
                "Failed to save Reddit results",
                execution_id=self.execution_id,
                agent_id=self.agent.id,
                error=str(e),
                exc_info=True
            )
            self.db_session.rollback()
            return False

    def _save_reddit_posts(self, posts: List[RedditPost]):
        """
        Save Reddit posts to the reddit_posts table.

        Args:
            posts: List of RedditPost objects from Reddit agent
        """
        if not posts:
            return

        # First, deduplicate within the current batch (keep first occurrence)
        seen_in_batch = set()
        unique_posts = []
        duplicates_in_batch = 0
        
        for post in posts:
            if post.post_id and post.post_id not in seen_in_batch:
                unique_posts.append(post)
                seen_in_batch.add(post.post_id)
            elif post.post_id in seen_in_batch:
                duplicates_in_batch += 1
                logger.debug(
                    "Duplicate post found in batch, skipping",
                    post_id=post.post_id
                )

        if duplicates_in_batch > 0:
            logger.info(
                "Deduplicated posts within batch",
                total_posts=len(posts),
                unique_posts=len(unique_posts),
                duplicates_removed=duplicates_in_batch
            )

        # Collect post IDs to check for existing posts in bulk
        post_ids = [post.post_id for post in unique_posts if post.post_id]
        existing_posts = set()
        
        if post_ids:
            existing_query = self.db_session.query(RedditPostModel.post_id).filter(
                RedditPostModel.post_id.in_(post_ids)
            )
            existing_posts = {row[0] for row in existing_query.all()}

        posts_to_add = []
        for post in unique_posts:
            post_id = post.post_id
            if not post_id or post_id in existing_posts:
                if post_id in existing_posts:
                    logger.debug(
                        "Reddit post already exists in database, skipping",
                        post_id=post_id
                    )
                continue

            try:
                # Parse created timestamp
                created_utc = None
                if post.created:
                    try:
                        created_utc = datetime.fromisoformat(post.created)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse created timestamp: {post.created}")

                # Create RedditPostModel instance
                reddit_post = RedditPostModel(
                    subreddit=post.subreddit,
                    post_id=post_id,
                    post_title=post.post_title,
                    post_body=post.post_body or "",
                    post_url=post.post_url,
                    created_utc=created_utc,
                    upvotes=post.upvotes or 0,
                    comment_count=post.comment_count or 0,
                    
                    # Combined text for embeddings/search (core post data)
                    text=post.text
                )

                posts_to_add.append(reddit_post)

            except Exception as e:
                logger.warning(
                    "Failed to prepare Reddit post for saving",
                    post_id=post_id,
                    error=str(e)
                )

        # Bulk add all new posts
        if posts_to_add:
            try:
                self.db_session.add_all(posts_to_add)
                logger.info(f"Prepared {len(posts_to_add)} Reddit posts for bulk insert")
            except Exception as e:
                logger.error(f"Failed to bulk add Reddit posts: {str(e)}")
                # Fallback to individual adds
                for post in posts_to_add:
                    try:
                        self.db_session.add(post)
                    except Exception as individual_error:
                        logger.warning(
                            "Failed to add individual Reddit post",
                            post_id=getattr(post, 'post_id', 'unknown'),
                            error=str(individual_error)
                        )

    def _save_reddit_execution_mappings(self, posts: List[RedditPost]):
        """
        Save Reddit execution mappings to track which posts were found in which execution.

        Args:
            posts: List of RedditPost objects from Reddit agent
        """
        logger.info(
            "Saving Reddit execution mappings",
            execution_id=self.execution_id,
            agent_id=self.agent.id,
            posts_count=len(posts) if posts else 0
        )
        if not posts:
            logger.info(
                "No posts found in Reddit execution mapping batch",
                execution_id=self.execution_id,
                agent_id=self.agent.id
            )
            return

        # First, deduplicate within the current batch (keep first occurrence)
        seen_in_batch = set()
        unique_posts = []
        duplicates_in_batch = 0
        
        for post in posts:
            if post.post_id and post.post_id not in seen_in_batch:
                unique_posts.append(post)
                seen_in_batch.add(post.post_id)
            elif post.post_id in seen_in_batch:
                duplicates_in_batch += 1
                logger.debug(
                    "Duplicate post found in execution mapping batch, skipping",
                    post_id=post.post_id
                )

        if duplicates_in_batch > 0:
            logger.info(
                "Deduplicated execution mappings within batch",
                total_posts=len(posts),
                unique_posts=len(unique_posts),
                duplicates_removed=duplicates_in_batch
            )

        # Collect existing mappings to avoid duplicates
        post_ids = [post.post_id for post in unique_posts if post.post_id]
        existing_mappings = set()
        
        if post_ids:
            existing_query = self.db_session.query(
                RedditAgentExecutionMapperModel.post_id
            ).filter(
                RedditAgentExecutionMapperModel.execution_id == self.execution_id,
                RedditAgentExecutionMapperModel.agent_id == self.agent.id,
                RedditAgentExecutionMapperModel.post_id.in_(post_ids)
            )
            existing_mappings = {row[0] for row in existing_query.all()}
        logger.info(f"Existing mappings: {len(existing_mappings)}")
        mappings_to_add = []
        for post in unique_posts:
            post_id = post.post_id
            if not post_id or post_id in existing_mappings:
                if post_id in existing_mappings:
                    logger.debug(
                        "Reddit execution mapping already exists in database, skipping",
                        execution_id=self.execution_id,
                        post_id=post_id
                    )
                continue

            try:
                # Determine primary relevance score (prefer combined, fallback to others)
                primary_relevance = (
                    getattr(post, 'combined_relevance', None) or 
                    getattr(post, 'semantic_relevance', None) or 
                    getattr(post, 'llm_relevance', None) or 
                    getattr(post, 'keyword_relevance', None) or 
                    getattr(post, 'relevance_score', None) or 
                    0.0
                )

                mapping = RedditAgentExecutionMapperModel(
                    execution_id=self.execution_id,
                    agent_id=self.agent.id,
                    post_id=post_id,
                    
                    # Relevance scores from agent analysis
                    relevance_score=primary_relevance,
                    keyword_relevance=getattr(post, 'keyword_relevance', None),
                    semantic_relevance=getattr(post, 'semantic_relevance', None),
                    combined_relevance=getattr(post, 'combined_relevance', None),
                    llm_relevance=getattr(post, 'llm_relevance', None),
                    
                    # Processing metadata
                    matched_query=getattr(post, 'matched_query', None),
                    sort_method=getattr(post, 'sort_method', None),
                    
                    # Agent output and processing
                    comment_draft=getattr(post, 'comment_draft', None),
                    status="processed"
                )

                mappings_to_add.append(mapping)

            except Exception as e:
                logger.warning(
                    "Failed to prepare Reddit execution mapping for saving",
                    execution_id=self.execution_id,
                    post_id=post_id,
                    error=str(e)
                )

        # Bulk add all new mappings
        if mappings_to_add:
            try:
                self.db_session.add_all(mappings_to_add)
                logger.info(f"Prepared {len(mappings_to_add)} Reddit execution mappings for bulk insert")
            except Exception as e:
                logger.error(f"Failed to bulk add Reddit execution mappings: {str(e)}")
                # Fallback to individual adds
                for mapping in mappings_to_add:
                    try:
                        self.db_session.add(mapping)
                    except Exception as individual_error:
                        logger.warning(
                            "Failed to add individual Reddit execution mapping",
                            execution_id=self.execution_id,
                            post_id=getattr(mapping, 'post_id', 'unknown'),
                            error=str(individual_error)
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
