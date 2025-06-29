from app.core.agents_tasks.rdagent.rd_agent_v1.state import AgentState
import structlog
from typing import Generator
logger = structlog.get_logger()


def create_OR_query_in_batch(query_list: list[str], batch_size: int) -> Generator[str, None, None]:
    for i in range(0, len(query_list), batch_size):
        batch = query_list[i:i+batch_size]
        query = " OR ".join(batch)
        yield query

def create_AND_query_in_batch(query_list: list[str], batch_size: int) -> Generator[str, None, None]:
    for i in range(0, len(query_list), batch_size):
        batch = query_list[i:i+batch_size]
        query = " AND ".join(batch)
        yield query

def create_NOT_query_in_batch(query_list: list[str], batch_size: int) -> Generator[str, None, None]:
    for i in range(0, len(query_list), batch_size):
        batch = query_list[i:i+batch_size]
        query = " NOT ".join(batch)
        yield query