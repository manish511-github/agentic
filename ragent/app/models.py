from sqlalchemy import Enum, Text
import enum
from datetime import datetime
import uuid
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Boolean, ARRAY, Index, UniqueConstraint, CheckConstraint


class Base(DeclarativeBase):
    pass

# ------------------------------------------------------------
# User and Auth Models
# ------------------------------------------------------------

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(100))
    is_active = Column(Boolean, default=False)
    verified_at = Column(DateTime, nullable=True, default=None)
    updated_at = Column(DateTime, nullable=True, default=None, onupdate=datetime.now)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    projects = relationship("ProjectModel", back_populates="owner")
    tokens = relationship("UserToken", back_populates="user")
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    
    def get_context_string(self, context: str):
        # Creating unique string
        return f"{context}{self.hashed_password[-6:]}{self.updated_at.strftime('%m%d%Y%H%M%S')}".strip()
    # create a unique index on username and email
    __table_args__ = (Index('idx_username_email', "username", "email"),)

class OAuthAccount(Base):
    __tablename__ = "oauth_accounts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider = Column(String(50), nullable=False)  # e.g., "google"
    provider_user_id = Column(String(255), nullable=False)  # e.g., Google "sub"
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expiry = Column(DateTime, nullable=True)
    scope = Column(ARRAY(String), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now()) 
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now()) 

    user = relationship("UserModel", back_populates="oauth_accounts")

class OAuthState(Base):
    __tablename__ = "oauth_states"
    id = Column(Integer, primary_key=True)
    state = Column(String(128), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class UserToken(Base):
    __tablename__ = "user_tokens"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    access_key = Column(String(250), nullable=True, index=True, default=None)
    refresh_key = Column(String(250), nullable=True, index=True, default=None)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    expires_at = Column(DateTime, nullable=False)

    user =relationship("UserModel", back_populates="tokens")

# ------------------------------------------------------------
# Project Model
# ------------------------------------------------------------
class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True,
                  default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(String)
    target_audience = Column(String)
    website_url = Column(String)
    category = Column(String)
    priority = Column(String)
    due_date = Column(DateTime, nullable=True)
    budget = Column(String, nullable=True)
    team = Column(JSON, nullable=True)
    tags = Column(String, nullable=True)
    competitors = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)
    excluded_keywords = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("UserModel", back_populates="projects")

    agents = relationship("AgentModel", back_populates="project")
    agent_results = relationship("AgentResultModel", back_populates="project")


class WebsiteDataModel(Base):
    __tablename__ = "website_data"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    title = Column(String)
    description = Column(String)
    target_audience = Column(String)
    keywords = Column(JSON)
    products_services = Column(JSON)
    main_category = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ------------------------------------------------------------
# Social Post related modal
# ------------------------------------------------------------


class RedditPostModel(Base):
    __tablename__ = "reddit_posts"
    id = Column(Integer, primary_key=True, index=True)
    subreddit = Column(String)
    post_id = Column(String, unique=True)
    post_title = Column(String)
    post_body = Column(String)
    post_url = Column(String)
    created_utc = Column(DateTime)
    upvotes = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # create an index on post_id
    __table_args__ = (Index('idx_post_id', "post_id", unique=True),)


class TwitterPostModel(Base):
    __tablename__ = "twitter_posts"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String)
    goals = Column(ARRAY(String))
    instructions = Column(String)
    tweet_id = Column(String)
    text = Column(String)
    created_at = Column(DateTime)
    user_name = Column(String)
    user_screen_name = Column(String)
    retweet_count = Column(Integer)
    favorite_count = Column(Integer)
    relevance_score = Column(Float)
    hashtags = Column(ARRAY(String))
    created = Column(DateTime, default=datetime.utcnow)

# ------------------------------------------------------------
# Agent related models
# ------------------------------------------------------------


class AgentModel(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String, nullable=False)
    description = Column(String)
    # reddit, twitter, linkedin
    agent_platform = Column(String, nullable=False)
    agent_status = Column(String, default="active")
    goals = Column(String)
    instructions = Column(String)
    expectations = Column(String)
    keywords = Column(ARRAY(String))
    project_id = Column(String, ForeignKey("projects.uuid"))
    mode = Column(String, default="copilot")  # copilot, autonomous
    review_minutes = Column(Integer, nullable=True)
    advanced_settings = Column(JSON, nullable=True)
    # Platform-specific settings
    platform_settings = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_run = Column(DateTime(timezone=True),
                      nullable=True)  # Track last run time
    oauth_account_id = Column(Integer, ForeignKey("oauth_accounts.id"), nullable=True)

    # Relationships
    project = relationship("ProjectModel", back_populates="agents")
    results = relationship("AgentResultModel", back_populates="agent")
    schedule = relationship(
        "ScheduleModel", back_populates="agent", uselist=False)
    executions = relationship("ExecutionModel", back_populates="agent")
    oauth_account = relationship("OAuthAccount")

# ------------------------------------------------------------
# Schedule and Execution Models
# ------------------------------------------------------------


class ScheduleTypeEnum(str, enum.Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class ExecutionStatusEnum(str, enum.Enum):
    scheduled = "scheduled"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ScheduleModel(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    schedule_type = Column(Enum(ScheduleTypeEnum), nullable=False)
    schedule_time = Column(DateTime, nullable=True)

    # Weekly schedule config
    days_of_week = Column(ARRAY(String), nullable=True)

    # Monthly schedule config
    day_of_month = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    agent = relationship("AgentModel", back_populates="schedule")
    executions = relationship("ExecutionModel", back_populates="schedule")

    __table_args__ = (
        # only one schedule should be active for an agent
        UniqueConstraint('agent_id', name='uix_agent_schedule'),
        # if schedule type is weekly the only days of week are valid
        CheckConstraint(
            "schedule_type != 'weekly' OR days_of_week IS NOT NULL", name='chk_days_of_week'),
        # if schedule type is monthly the day of month is valid
        CheckConstraint(
            "schedule_type != 'monthly' OR day_of_month IS NOT NULL", name='chk_day_of_month'),
    )


class ExecutionModel(Base):
    __tablename__ = "executions"
    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, ForeignKey("schedules.id"))
    agent_id = Column(Integer, ForeignKey("agents.id"))
    schedule_time = Column(DateTime, nullable=False)
    status = Column(Enum(ExecutionStatusEnum), nullable=False)
    results = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    agent = relationship("AgentModel", back_populates="executions")
    schedule = relationship("ScheduleModel", back_populates="executions")
    reddit_agent_execution_mapper = relationship(
        "RedditAgentExecutionMapperModel", back_populates="execution")


class RedditAgentExecutionMapperModel(Base):
    __tablename__ = "reddit_agent_execution_mapper"
    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(Integer, ForeignKey("executions.id"))
    agent_id = Column(Integer, ForeignKey("agents.id"))
    post_id = Column(String, ForeignKey("reddit_posts.post_id"))
    relevance_score = Column(Float)
    comment_draft = Column(String)
    status = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    execution = relationship(
        "ExecutionModel", back_populates="reddit_agent_execution_mapper")

    # create a unique index on execution_id and agent_id
    __table_args__ = (Index('idx_execution_agent',
                      "execution_id", "agent_id"),)


class AgentResultModel(Base):
    __tablename__ = "agent_results"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    project_id = Column(String, ForeignKey("projects.uuid"))
    results = Column(JSON)
    status = Column(String)  # completed, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    agent = relationship("AgentModel", back_populates="results")
    project = relationship("ProjectModel", back_populates="agent_results")

