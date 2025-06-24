"""
Agent Executor System

This package provides a clean, extensible system for executing different types
of marketing agents (Reddit, Hacker News, etc.) in a Celery worker environment.

Main Components:
- BaseAgentExecutor: Abstract base class for all executors
- ExecutorFactory: Factory for creating appropriate executors
- ResultMapper: System for saving agent results to database
- Specific Executors: Reddit, Hacker News, etc.

Usage:
    from app.services.executors import run_agent
    
    # This will be called by Celery scheduler
    result = run_agent.delay(execution_id, agent_id)
"""

from .executor import run_agent
from .base_executor import BaseAgentExecutor
from .executor_factory import ExecutorFactory
from .reddit_executor import RedditAgentExecutor
from .hackernews_executor import HackerNewsAgentExecutor
from .result_mappers import (
    ResultMapper,
    RedditResultMapper,
    HackerNewsResultMapper,
    get_result_mapper
)

__all__ = [
    "run_agent",
    "BaseAgentExecutor",
    "ExecutorFactory",
    "RedditAgentExecutor",
    "HackerNewsAgentExecutor",
    "ResultMapper",
    "RedditResultMapper",
    "HackerNewsResultMapper",
    "get_result_mapper"
]
