from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, TypedDict

class HNComment(BaseModel):
    id: int
    text: Optional[str]
    sentiment: Optional[str] = None
    relevance: Optional[float] = None
    parent_story_id: Optional[int] = None
    story_id: Optional[int] = None
    children: List[int] = []

class HNStory(BaseModel):
    id: int
    title: str
    text: Optional[str]
    url: Optional[str]
    score: int
    time: int
    relevance: float
    summary: Optional[str] = None
    top_comments: Optional[List[HNComment]] = None
    story_id: Optional[int] = None
    children: List[int] = []
    relevant_comment_ids: List[int] = []

class HNAgentInput(BaseModel):
    agent_name: str = Field(..., min_length=1)
    goals: List[str]
    instructions: str
    company_keywords: List[str]
    description: Optional[str] = None
    expectation: str
    target_audience: Optional[str] = None
    min_score: Optional[int] = 10
    max_age_days: Optional[int] = 7

class HNAgentOutput(BaseModel):
    agent_name: str
    goals: List[str]
    instructions: str
    stories: List[Any]

class AgentState(TypedDict):
    agent_input: HNAgentInput
    collection_name: str
    expanded_queries: List[str]
    raw_search_results: List[Any]
    found_stories_map: Dict[int, HNStory]
    relevant_comments_found: List[HNComment]
    final_stories_output: List[HNStory] 