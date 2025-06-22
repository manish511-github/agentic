from pydantic import BaseModel, Field
from typing import List, Dict, Optional, TypedDict, Any
from sqlalchemy.ext.asyncio import AsyncSession

class TwitterAgentInput(BaseModel):
    agent_name: str
    goals: List[str]
    instructions: str
    description: str
    expectation: str
    target_audience: str
    company_keywords: List[str]
    min_likes: int = Field(default=10)
    max_age_days: int = Field(default=7)

class TweetQueryInput(BaseModel):
    query: str
    minimum_tweets: int = Field(default=10)
    product: str = Field(default="Top")
    output_file: Optional[str] = None

class AgentState(TypedDict):
    agent_name: str
    goals: List[str]
    instructions: str
    description: str
    expectation: str
    target_audience: str
    company_keywords: List[str]
    min_likes: int
    max_age_days: int
    hashtags: List[str]
    tweets: List[Dict]
    retries: int
    error: Optional[str]
    db: AsyncSession

class TwitterAgentState(TypedDict):
    agent_input: TwitterAgentInput
    hashtags: List[str]
    tweets: List[Dict[str, Any]]
    collection_name: str
    vector_db_results: List[Any]
    semantic_results: List[Any]
    error: Optional[str]
    db: Any 