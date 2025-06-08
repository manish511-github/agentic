from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class ProjectBase(BaseModel):
    title: str
    description: Optional[str] = None
    target_audience: Optional[str] = None
    website_url: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[datetime] = None
    budget: Optional[str] = None
    team: Optional[List[Dict]] = None
    tags: Optional[str] = None
    competitors: Optional[List[Dict]] = None
    keywords: Optional[List[str]] = None
    excluded_keywords: Optional[List[str]] = None

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    uuid: str
    owner_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class RedditSettings(BaseModel):
    subreddit: str
    timeRange: str  # day, week, month, year, all
    relevanceThreshold: int
    monitorComments: bool
    minUpvotes: int
    excluded_keywords: Optional[List[str]] = None
    target_subreddits: Optional[List[str]] = None

class TwitterSettings(BaseModel):
    target_accounts: List[str]
    timeRange: str  # day, week, month
    relevanceThreshold: int
    sentiment_filter: Optional[str] = None  # positive, negative, neutral
    excluded_keywords: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None

class LinkedInSettings(BaseModel):
    target_companies: List[str]
    timeRange: str  # day, week, month
    relevanceThreshold: int
    industry_filter: Optional[List[str]] = None
    excluded_keywords: Optional[List[str]] = None
    job_titles: Optional[List[str]] = None

class PlatformSettings(BaseModel):
    reddit: Optional[RedditSettings] = None
    twitter: Optional[TwitterSettings] = None
    linkedin: Optional[LinkedInSettings] = None

class AgentBase(BaseModel):
    agent_name: str
    agent_platform: str
    agent_status: str = "active"
    goals: str
    instructions: str
    expectations: str
    mode: str = "copilot"
    review_period: str
    review_minutes: int = 0
    advanced_settings: Dict = {}
    platform_settings: PlatformSettings

class AgentCreate(AgentBase):
    project_id: str

class Agent(AgentBase):
    id: int
    project_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class RedditPost(BaseModel):
    subreddit: str
    post_id: str
    post_title: str
    post_body: str
    post_url: str
    relevance_score: float
    sentiment_score: Optional[float] = None
    comment_draft: Optional[str] = None
    status: Optional[str] = None
    created_at: str

    class Config:
        orm_mode = True

class AgentResult(BaseModel):
    id: int
    agent_id: int
    project_id: int
    status: str
    results: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime

    class Config:
        orm_mode = True
