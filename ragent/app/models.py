from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Boolean, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

class Base(DeclarativeBase):
    pass

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

class RedditPostModel(Base):
    __tablename__ = "reddit_posts"
    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String, nullable=False)
    goals = Column(JSON)
    instructions = Column(String)
    subreddit = Column(String)
    post_id = Column(String)
    post_title = Column(String)
    post_body = Column(String)
    post_url = Column(String)
    comment_draft = Column(String)
    status = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    upvotes = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    created = Column(DateTime(timezone=True))
    keyword_relevance = Column(Float)
    matched_query = Column(String)
    semantic_relevance = Column(Float)
    combined_relevance = Column(Float)

class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
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

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    projects = relationship("ProjectModel", back_populates="owner")

class AgentModel(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String, nullable=False)
    agent_platform = Column(String, nullable=False)  # reddit, twitter, linkedin
    agent_status = Column(String, default="active")
    goals = Column(String)
    instructions = Column(String)
    expectations = Column(String)
    project_id = Column(Integer, ForeignKey("projects.id"))
    mode = Column(String, default="copilot")  # copilot, autonomous
    review_period = Column(String)  # daily, weekly, monthly
    review_minutes = Column(Integer, default=0)
    advanced_settings = Column(JSON, default={})
    platform_settings = Column(JSON)  # Platform-specific settings
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_run = Column(DateTime(timezone=True), nullable=True)  # Track last run time

    # Relationships
    project = relationship("ProjectModel", back_populates="agents")
    results = relationship("AgentResultModel", back_populates="agent")

class AgentResultModel(Base):
    __tablename__ = "agent_results"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    project_id = Column(Integer, ForeignKey("projects.id"))
    results = Column(JSON)
    status = Column(String)  # completed, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    agent = relationship("AgentModel", back_populates="results")
    project = relationship("ProjectModel", back_populates="agent_results")

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