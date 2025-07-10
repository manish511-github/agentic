from pydantic import BaseModel, model_validator
from typing import List, Dict, Optional
from datetime import datetime
import enum


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class User(BaseModel):
    id: int
    username: str
    email: str

    model_config = {"from_attributes": True}


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

    model_config = {"from_attributes": True}


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

    model_config = {"from_attributes": True}


class ScheduleTypeEnum(str, enum.Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class DaysOfWeekEnum(str, enum.Enum):
    monday = "monday"
    tuesday = "tuesday"
    wednesday = "wednesday"
    thursday = "thursday"
    friday = "friday"
    saturday = "saturday"


class ScheduleBase(BaseModel):
    schedule_type: ScheduleTypeEnum
    schedule_time: Optional[datetime] = None
    days_of_week: Optional[List[DaysOfWeekEnum]] = []
    day_of_month: Optional[int] = None


class RedditSettings(BaseModel):
    subreddits: Optional[List[str]] = None
    timeRange: Optional[str] = None  # day, week, month, year, all
    relevanceThreshold: Optional[int] = None
    monitorComments: Optional[bool] = False
    minUpvotes: Optional[int] = None
    excluded_keywords: Optional[List[str]] = None


class TwitterSettings(BaseModel):
    target_accounts: Optional[List[str]] = None
    timeRange: Optional[str] = None  # day, week, month
    relevanceThreshold: Optional[int] = None
    sentiment_filter: Optional[str] = None  # positive, negative, neutral
    excluded_keywords: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None


class LinkedInSettings(BaseModel):
    target_companies: Optional[List[str]] = None
    timeRange: Optional[str] = None  # day, week, month
    relevanceThreshold: Optional[int] = None
    industry_filter: Optional[List[str]] = None
    excluded_keywords: Optional[List[str]] = None
    job_titles: Optional[List[str]] = None


class PlatformSettings(BaseModel):
    reddit: Optional[RedditSettings] = None
    twitter: Optional[TwitterSettings] = None
    linkedin: Optional[LinkedInSettings] = None


class AgentPlatformEnum(str, enum.Enum):
    reddit = "reddit"
    twitter = "twitter"
    linkedin = "linkedin"


class AgentModeEnum(str, enum.Enum):
    copilot = "copilot"
    autonomous = "autonomous"


class AgentBase(BaseModel):
    agent_name: str
    description: Optional[str] = None
    agent_platform: AgentPlatformEnum
    agent_status: str = "active"
    goals: str
    instructions: str
    expectations: str
    agent_keywords: List[str]
    project_id: str
    mode: AgentModeEnum
    review_minutes: Optional[int] = None
    oauth_account_id: Optional[int] = None
    advanced_settings: Optional[Dict] = None
    platform_settings: Optional[Dict] = None


class AgentCreate(AgentBase):
    schedule: Optional[ScheduleBase] = None


class Agent(AgentBase):
    id: int
    created_at: datetime
    last_run: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AgentResult(BaseModel):
    id: int
    agent_id: int
    project_id: str  # Changed to str since it's a UUID
    status: str
    results: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
