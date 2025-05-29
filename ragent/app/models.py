from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.orm import DeclarativeBase
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

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)