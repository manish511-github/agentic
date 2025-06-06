import math
from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from typing import List, Dict, Optional, TypedDict
import os
import random
import hashlib

from dotenv import load_dotenv
from urllib.parse import urljoin
import re
from tenacity import retry, stop_after_attempt, wait_fixed,wait_exponential
import structlog
import json
import asyncpraw
import asyncprawcore  # Import asyncprawcore to handle its exceptions
from datetime import datetime, timedelta
import asyncio
from app.models import WebsiteDataModel, RedditPostModel, AgentModel, AgentResultModel
from app.database import get_db
from functools import lru_cache
from sqlalchemy.orm import Session

# Initialize logging
import logging

# Standard logging configuration
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)

# Structlog configuration
structlog.configure(
    processors=[
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

# Initialize logger
logger = structlog.get_logger()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

# FastAPI router
router = APIRouter()

class AgentState(TypedDict):
    agent_name: str
    goals: List[str]
    description: str
    expectation: str
    target_audience: str
    instructions: str
    company_keywords: List[str]
    min_upvotes: int
    max_age_days: int
    restrict_to_goal_subreddits: bool
    subreddits: List[str]
    posts: List[Dict]
    retries: int
    error: Optional[str]
    db: Session

class RedditAgentInput(BaseModel):
    agent_name: str
    goals: List[str]
    instructions: str
    description: str
    expectation: str
    target_audience: str
    company_keywords: List[str]
    min_upvotes: int = 0
    max_age_days: int = 7
    restrict_to_goal_subreddits: bool = False

async def get_reddit_client():
    """Get Reddit client"""
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD
    )
    return reddit

def validate_input_node(state: AgentState) -> AgentState:
    """Validate input parameters"""
    valid_goals = ["increase brand awareness", "engage potential customers", "grow web traffic"]
    if not all(goal.lower() in valid_goals for goal in state["goals"]):
        state["error"] = f"Invalid goals. Choose from: {valid_goals}"
        return state
    state["retries"] = 0
    state["subreddits"] = []
    state["posts"] = []
    logger.info("Input validated", agent_name=state["agent_name"])
    return state

def search_subreddits_node(state: AgentState) -> AgentState:
    """Search for relevant subreddits synchronously"""
    if state.get("error"):
        return state
    
    try:
        # For now, just use predefined subreddits
        goal_subreddits = {
            "increase brand awareness": ["marketing", "Branding", "Advertising"],
            "engage potential customers": ["AskReddit", "smallbusiness", "Entrepreneur"],
            "grow web traffic": ["SEO", "DigitalMarketing", "growthhacking"]
        }
        
        target_subreddits = set()
        if state["restrict_to_goal_subreddits"]:
            for goal in state["goals"]:
                target_subreddits.update(goal_subreddits.get(goal.lower(), []))
        
        state["subreddits"] = list(target_subreddits)[:5]  # Limit to 5 subreddits
        logger.info("Subreddits selected", agent_name=state["agent_name"], subreddits=state["subreddits"])
        
    except Exception as e:
        state["error"] = f"Subreddit search failed: {str(e)}"
        logger.error("Subreddit search failed", agent_name=state["agent_name"], error=str(e))
    
    return state

def fetch_posts_node(state: AgentState) -> AgentState:
    """Fetch posts synchronously"""
    if state.get("error"):
        return state
    
    try:
        # For now, just add a dummy post
        state["posts"] = [{
            "subreddit": "test",
            "post_id": "123",
            "post_title": "Test Post",
            "post_body": "This is a test post",
            "post_url": "https://reddit.com/r/test/123",
            "relevance_score": 0.8
        }]
        logger.info("Fetched posts", agent_name=state["agent_name"], post_count=len(state["posts"]))
        
    except Exception as e:
        state["error"] = f"Post fetching failed: {str(e)}"
        logger.error("Fetch failure", error=str(e), agent_name=state["agent_name"])
    
    return state

async def calculate_post_relevance(text: str, keywords: List[str], expectation: str) -> float:
    """Calculate relevance score for a post"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            max_retries=2,
            timeout=30
        )
        
        prompt = PromptTemplate(
            input_variables=["text", "keywords", "expectation"],
            template=(
                "Evaluate the relevance of this text to our marketing needs.\n\n"
                "TEXT:\n{text}\n\n"
                "KEYWORDS:\n{keywords}\n\n"
                "EXPECTATION:\n{expectation}\n\n"
                "Rate the relevance from 0.0 to 1.0, where:\n"
                "- 0.0 means completely irrelevant\n"
                "- 1.0 means perfectly relevant\n\n"
                "Return ONLY the number, no explanation or formatting.\n"
                "Example: 0.85"
            )
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        result = await chain.arun(
            text=text,
            keywords=", ".join(keywords),
            expectation=expectation
        )
        
        try:
            score = float(result.strip())
            return min(max(score, 0.0), 1.0)
        except ValueError:
            logger.warning("Failed to parse relevance score", result=result)
            return 0.0
            
    except Exception as e:
        logger.warning("Relevance calculation failed", error=str(e))
        return 0.0

def store_results_node(state: AgentState) -> AgentState:
    """Store results in database synchronously"""
    if state.get("error"):
        return state
        
    try:
        db = state["db"]
        for post in state["posts"]:
            reddit_post = RedditPostModel(
                agent_name=state["agent_name"],
                goals=",".join(state["goals"]),
                instructions=state["instructions"],
                subreddit=post["subreddit"],
                post_id=post["post_id"],
                post_title=post["post_title"],
                post_body=post["post_body"],
                post_url=post["post_url"],
                relevance_score=post["relevance_score"],
                created_at=datetime.utcnow()
            )
            db.add(reddit_post)
        db.commit()
            
        logger.info(
            "Stored Reddit posts",
            agent_name=state["agent_name"],
            post_count=len(state["posts"])
        )
        
    except Exception as e:
        state["error"] = f"Database error: {str(e)}"
        logger.error("Database error", agent_name=state["agent_name"], error=str(e))
        
    return state

def create_workflow_graph():
    """Create the workflow graph"""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("search_subreddits", search_subreddits_node)
    graph.add_node("fetch_posts", fetch_posts_node)
    graph.add_node("store_results", store_results_node)
    
    # Set entry point and edges
    graph.set_entry_point("validate_input")
    graph.add_edge("validate_input", "search_subreddits")
    graph.add_edge("search_subreddits", "fetch_posts")
    graph.add_edge("fetch_posts", "store_results")
    graph.add_edge("store_results", END)
    
    return graph.compile()

# Create the graph once at module level
reddit_graph = create_workflow_graph()

def run_sync_workflow(state: dict) -> dict:
    """Run the workflow synchronously (for Celery)"""
    try:
        # Create a synchronous version of the workflow
        sync_graph = StateGraph(AgentState)
        
        # Add synchronous versions of the nodes
        sync_graph.add_node("validate_input", validate_input_node)
        sync_graph.add_node("search_subreddits", search_subreddits_node)
        sync_graph.add_node("fetch_posts", fetch_posts_node)
        sync_graph.add_node("store_results", store_results_node)
        
        # Set entry point and edges
        sync_graph.set_entry_point("validate_input")
        sync_graph.add_edge("validate_input", "search_subreddits")
        sync_graph.add_edge("search_subreddits", "fetch_posts")
        sync_graph.add_edge("fetch_posts", "store_results")
        sync_graph.add_edge("store_results", END)
        
        # Run the workflow synchronously
        compiled_graph = sync_graph.compile()
        result = compiled_graph.invoke(state)
        
        # Ensure any pending database operations are committed
        if result.get("db"):
            result["db"].commit()
        
        return result
        
    except Exception as e:
        logger.error("Sync workflow failed", error=str(e))
        state["error"] = str(e)
        return state

@router.post("/reddit/reddit-agent", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def run_reddit_agent(input: RedditAgentInput, db: AsyncSession = Depends(get_db)):
    """Run the Reddit agent workflow"""
    try:
        initial_state = AgentState(
            agent_name=input.agent_name,
            goals=input.goals,
            instructions=input.instructions,
            description=input.description,
            expectation=input.expectation,
            target_audience=input.target_audience,
            company_keywords=input.company_keywords,
            min_upvotes=input.min_upvotes,
            max_age_days=input.max_age_days,
            restrict_to_goal_subreddits=input.restrict_to_goal_subreddits,
            subreddits=[],
            posts=[],
            retries=0,
            error=None,
            db=db
        )
        
        result = await reddit_graph.ainvoke(initial_state)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {
            "status": "completed",
            "subreddits": result["subreddits"],
            "posts": result["posts"]
        }
        
    except Exception as e:
        logger.error("Reddit agent processing failed", agent_name=input.agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
