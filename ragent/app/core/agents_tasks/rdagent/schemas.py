from typing import List
from pydantic import BaseModel, Field


class RedditPost(BaseModel):
    """Lightweight representation (logged / returned) of a Reddit submission."""

    subreddit: str = Field(...,
                           description="Subreddit name without the r/ prefix")
    post_id: str = Field(..., description="Reddit submission id")
    post_title: str = Field(..., description="Submission title")
    post_body: str = Field(...,
                           description="Submission self-text (may be empty)")
    post_url: str = Field(..., description="Canonical Reddit URL")
    relevance_score: float = Field(..., ge=0, le=1)
    sentiment_score: float = Field(..., ge=-1, le=1)


class RedditAgentOutput(BaseModel):
    """Final structured response returned by the Reddit agent."""

    agent_name: str
    goals: List[str]
    instructions: str
    posts: List[RedditPost]
