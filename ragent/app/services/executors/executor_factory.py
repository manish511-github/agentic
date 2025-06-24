from typing import Type
import structlog

from .base_executor import BaseAgentExecutor
from .reddit_executor import RedditAgentExecutor
from .hackernews_executor import HackerNewsAgentExecutor

logger = structlog.get_logger()


class ExecutorFactory:
    """
    Factory class for creating appropriate agent executors based on platform.

    This factory pattern allows for easy extension when new agent types are added.
    """

    # Registry of platform -> executor class mappings
    _executors = {
        "reddit": RedditAgentExecutor,
        "hackernews": HackerNewsAgentExecutor,
    }

    @classmethod
    def create_executor(cls, platform: str, execution_id: int, agent_id: int) -> BaseAgentExecutor:
        """
        Create the appropriate executor for the given platform.

        Args:
            platform: Agent platform (e.g., 'reddit', 'hackernews')
            execution_id: Execution ID to process
            agent_id: Agent ID to execute

        Returns:
            BaseAgentExecutor: Appropriate executor instance

        Raises:
            ValueError: If platform is not supported
        """
        platform = platform.lower()

        executor_class = cls._executors.get(platform)
        if not executor_class:
            supported_platforms = list(cls._executors.keys())
            raise ValueError(
                f"Unsupported platform: {platform}. "
                f"Supported platforms: {', '.join(supported_platforms)}"
            )

        logger.info(
            "Creating executor",
            platform=platform,
            execution_id=execution_id,
            agent_id=agent_id,
            executor_class=executor_class.__name__
        )

        return executor_class(execution_id, agent_id)

    @classmethod
    def register_executor(cls, platform: str, executor_class: Type[BaseAgentExecutor]):
        """
        Register a new executor for a platform.

        This allows for dynamic registration of new executors without modifying
        the factory class itself.

        Args:
            platform: Platform name
            executor_class: Executor class that extends BaseAgentExecutor
        """
        if not issubclass(executor_class, BaseAgentExecutor):
            raise ValueError(
                f"Executor class must extend BaseAgentExecutor, "
                f"got {executor_class.__name__}"
            )

        cls._executors[platform.lower()] = executor_class

        logger.info(
            "Registered new executor",
            platform=platform,
            executor_class=executor_class.__name__
        )

    @classmethod
    def get_supported_platforms(cls) -> list[str]:
        """
        Get list of supported platforms.

        Returns:
            list[str]: List of supported platform names
        """
        return list(cls._executors.keys())
