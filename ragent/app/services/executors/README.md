# Agent Executor System

## Overview

The Agent Executor System is a clean, extensible architecture for executing different types of marketing agents (Reddit, Hacker News, etc.) in a Celery worker environment. It follows object-oriented design principles and patterns to ensure maintainability and easy extension.

## Architecture

### Core Components

1. **BaseAgentExecutor** - Abstract base class providing common functionality
2. **ExecutorFactory** - Factory pattern for creating appropriate executors  
3. **ResultMapper** - Abstract mapper system for saving agent results
4. **Specific Executors** - Platform-specific implementations

### Design Patterns Used

- **Factory Pattern**: `ExecutorFactory` creates appropriate executors
- **Template Method**: `BaseAgentExecutor` defines the execution flow
- **Strategy Pattern**: Different result mappers for different platforms
- **Context Manager**: Proper resource management with database sessions

## File Structure

```
executors/
├── __init__.py                 # Package initialization and exports
├── README.md                   # This documentation
├── base_executor.py           # Abstract base executor class
├── executor_factory.py        # Factory for creating executors  
├── executor.py                # Main Celery task and orchestration
├── result_mappers.py          # Result mapping system
├── reddit_executor.py         # Reddit-specific executor
└── hackernews_executor.py     # Hacker News-specific executor
```

## Usage

### Basic Usage

The system is designed to be used by the Celery scheduler:

```python
from app.services.executors import run_agent

# Called by Celery scheduler
result = run_agent.delay(execution_id, agent_id)
```

### Adding New Agent Types

To add support for a new agent platform:

1. **Create a new executor class**:

```python
# twitter_executor.py
from .base_executor import BaseAgentExecutor
from .result_mappers import get_result_mapper
from app.twitter_agent import twitter_graph  # Your agent implementation

class TwitterAgentExecutor(BaseAgentExecutor):
    def create_initial_state(self) -> Dict[str, Any]:
        # Create initial state for Twitter agent
        return {
            "agent_name": self.agent.agent_name,
            "goals": self.agent.goals.split(",") if self.agent.goals else [],
            # ... other Twitter-specific fields
        }
    
    async def execute_agent(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        # Execute Twitter agent workflow
        return await twitter_graph.ainvoke(initial_state)
    
    def save_results(self, results: Dict[str, Any]) -> bool:
        # Use Twitter result mapper
        mapper = get_result_mapper("twitter", self.db_session, self.execution_id, self.agent)
        return mapper.save_results(results)
```

2. **Create a result mapper**:

```python
# Add to result_mappers.py
class TwitterResultMapper(ResultMapper):
    def save_results(self, results: Dict[str, Any]) -> bool:
        # Save Twitter-specific results
        # Save general agent results
        if not self.save_agent_result(results):
            return False
            
        # Save Twitter-specific data (tweets, etc.)
        tweets = results.get("tweets", [])
        # ... save tweets to database
        
        self.db_session.commit()
        return True
```

3. **Register the executor**:

```python
# In executor_factory.py, add to _executors dict:
_executors = {
    "reddit": RedditAgentExecutor,
    "hackernews": HackerNewsAgentExecutor,
    "twitter": TwitterAgentExecutor,  # Add this line
}

# Or register dynamically:
ExecutorFactory.register_executor("twitter", TwitterAgentExecutor)
```

4. **Update result mapper factory**:

```python
# In result_mappers.py, add to get_result_mapper function:
mappers = {
    "reddit": RedditResultMapper,
    "hackernews": HackerNewsResultMapper,
    "twitter": TwitterResultMapper,  # Add this line
}
```

## Configuration

### Database Models Required

Your agent platform should have these relationships in the database:

- `AgentModel` with `agent_platform` field
- `AgentModel` with relationship to `ProjectModel`
- Optional platform-specific result tables (like `RedditPostModel`)

### Agent Workflow Requirements

Your agent workflow should:

- Accept an initial state dictionary
- Return a results dictionary  
- Handle errors gracefully by returning `{"error": "message"}`
- Be async-compatible

## Error Handling

The system includes comprehensive error handling:

- **Database errors**: Automatic rollback and logging
- **Agent execution errors**: Captured and logged with full context
- **Validation errors**: Clear error messages for missing data
- **Retry logic**: Automatic retries for transient failures

## Logging

Structured logging is used throughout the system with contextual information:

```python
logger.info(
    "Agent execution completed",
    execution_id=execution_id,
    agent_id=agent_id,
    platform=agent.agent_platform,
    posts_found=len(results.get("posts", []))
)
```

## Best Practices

### Executor Implementation

1. **Keep executors focused**: Each executor should handle only one platform
2. **Use proper logging**: Include relevant context in all log messages
3. **Handle errors gracefully**: Return error dictionaries instead of raising exceptions
4. **Validate inputs**: Check that required data is present before execution

### Result Mapping

1. **Check for duplicates**: Avoid saving duplicate data to the database
2. **Use transactions**: Wrap database operations in transactions
3. **Handle partial failures**: Log warnings for individual item failures
4. **Clean data**: Remove non-serializable items before saving

### Testing

1. **Unit tests**: Test each executor and mapper independently
2. **Integration tests**: Test the full flow from Celery task to database
3. **Mock external services**: Use mocks for Reddit API, etc.
4. **Error scenarios**: Test error handling paths

## Monitoring

The system provides metrics for monitoring:

- Execution success/failure rates
- Processing times per platform
- Database save success rates
- Agent-specific metrics (posts found, stories processed, etc.)

## Performance Considerations

- **Async execution**: All agent workflows run asynchronously
- **Database sessions**: Proper session management with context managers  
- **Resource cleanup**: Automatic cleanup of resources on success/failure
- **Memory management**: Celery worker recycling after processing tasks

## Security

- **Input validation**: All inputs are validated before processing
- **SQL injection protection**: Using SQLAlchemy ORM prevents SQL injection
- **Error information**: Sensitive information is not exposed in error messages
- **Access control**: Agents can only access their associated project data 