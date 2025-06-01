from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func

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
    relevance_score = Column(Float)
    sentiment_score = Column(Float)
    comment_draft = Column(String)
    status = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
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

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    projects = relationship("ProjectModel", back_populates="owner")