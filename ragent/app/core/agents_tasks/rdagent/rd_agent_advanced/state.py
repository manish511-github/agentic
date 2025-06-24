from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentState(TypedDict):
    agent_name: str
    goals: List[str]
    description: str
    expectation: str
    target_audience: str
    instructions: str
    company_keywords: List[str]
    keywords: Optional[List[str]]
    min_upvotes: int
    max_age_days: int
    restrict_to_goal_subreddits: bool
    subreddits: List[str]
    posts: List[Dict]
    retries: int
    error: Optional[str]
    db: Optional[AsyncSession]
    llm: Optional[ChatGoogleGenerativeAI]

class RedditPost(BaseModel):
    subreddit: str
    post_id: str
    post_title: str
    post_body: str
    post_url: str
    relevance_score: float
    sentiment_score: float

class RedditAgentOutput(BaseModel):
    agent_name: str
    goals: List[str]
    instructions: str
    posts: List[RedditPost]

class RedditAgentInput(BaseModel):
    agent_name: str = Field(..., min_length=1, description="Name of the marketing agent")
    goals: List[str] = Field(
        ...,
        description="Goals for the agent (e.g., increase brand awareness, engage potential customers, grow web traffic)",
        examples=[["increase brand awareness", "engage potential customers"]]
    )
    instructions: str = Field(
        ...,
        description="Custom instructions for the agent",
        examples=["Focus on SaaS communities and promote our CRM product."]
    )
    company_keywords: List[str] = Field(
        ...,
        description="Keywords related to the company or product",
        examples=[["CRM", "SaaS", "customer management"]]
    )
    description: Optional[str] = Field(
        None,
        description="Brief description of the company or product"
    )
    expectation: str = Field(
        ...,
        description="Detailed expectation for post relevance",
        examples=["Posts about affordable SaaS CRM tools for small businesses with positive user feedback."]
    )
    target_audience: Optional[str] = Field(
        None,
        description="The primary target audience for the product or campaign"
    )
    min_upvotes: Optional[int] = Field(0, description="Minimum upvotes for posts")
    max_age_days: Optional[int] = Field(7, description="Maximum age of posts in days")
    restrict_to_goal_subreddits: Optional[bool] = Field(False, description="Restrict to predefined goal subreddits only") 