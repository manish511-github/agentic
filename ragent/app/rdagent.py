from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
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
from dotenv import load_dotenv
from urllib.parse import urljoin
import re
from tenacity import retry, stop_after_attempt, wait_fixed
import structlog
import json
import asyncpraw
import asyncprawcore  # Import asyncprawcore to handle its exceptions
from datetime import datetime, timedelta
import asyncio
from app.models import WebsiteDataModel, RedditPostModel
from app.database import get_db

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
    instructions: str
    company_keywords: List[str]
    min_upvotes: int
    max_age_days: int
    restrict_to_goal_subreddits: bool
    subreddits: List[str]
    posts: List[Dict]
    retries: int
    error: Optional[str]
    db: Optional[AsyncSession]  # Added to store db session

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
    min_upvotes: Optional[int] = Field(0, description="Minimum upvotes for posts")
    max_age_days: Optional[int] = Field(7, description="Maximum age of posts in days")
    restrict_to_goal_subreddits: Optional[bool] = Field(False, description="Restrict to predefined goal subreddits only")


async def get_reddit_client():
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD
    )
    return reddit

async def analyze_sentiment(content: str) -> float:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        sentiment_prompt = PromptTemplate(
            input_variables=["content"],
            template="Analyze the sentiment of the following text and return a score between -1 (negative) and 1 (positive): {content}"
        )
        sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
        score = float(await sentiment_chain.arun(content=content))
        return max(min(score, 1.0), -1.0)
    except Exception as e:
        logger.warning("Sentiment analysis failed", error=str(e))
        return 0.0
    
async def validate_input_node(state: AgentState) -> AgentState:
    valid_goals = ["increase brand awareness", "engage potential customers", "grow web traffic"]
    if not all(goal.lower() in valid_goals for goal in state["goals"]):
        state["error"] = f"Invalid goals. Choose from: {valid_goals}"
        return state
    state["retries"] = 0
    state["subreddits"] = []
    state["posts"] = []
    logger.info("Input validated", agent_name=state["agent_name"])
    return state


async def score_posts_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    try:
        scored_posts = []
        company_keywords = [kw.lower() for kw in state["company_keywords"]]
        
        # Filter posts that meet minimum relevance first (without sentiment)
        filtered_posts = []
        for post in state["posts"]:
            content = f"{post['post_title']} {post['post_body']}".lower()
            matches = sum(1 for kw in company_keywords if kw in content)
            relevance_score = matches / max(len(company_keywords), 1)
            if relevance_score > 0.0:
                print(relevance_score)
                # print(content)
                post["relevance_score"] = relevance_score
                filtered_posts.append((post, content))
        # Prepare coroutines for sentiment analysis
        # sentiment_tasks = [analyze_sentiment(content) for _, content in filtered_posts]
        # sentiment_scores = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
        
        # for (post, _), sentiment_score in zip(filtered_posts, sentiment_scores):
        #     # Handle possible exceptions from analyze_sentiment
        #     if isinstance(sentiment_score, Exception):
        #         logger.warning("Sentiment analysis failed for a post", error=str(sentiment_score))
        #         continue
        #     if sentiment_score >= 0:
        #         post["sentiment_score"] = sentiment_score
        #         scored_posts.append(post)

        # Sort and keep top 10 by relevance
        # state["posts"] = sorted(scored_posts, key=lambda x: x["relevance_score"], reverse=True)[:10]
        import pprint
        pprint.pprint(filtered_posts)
        logger.info("Posts scored", agent_name=state["agent_name"], post_count=len(state["posts"]))
    except Exception as e:
        state["error"] = f"Post scoring failed: {str(e)}"
        logger.error("Post scoring failed", agent_name=state["agent_name"], error=str(e))
    return state

async def search_subreddits_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    try:
        async with await get_reddit_client() as reddit:
            goal_subreddits = {
                "increase brand awareness": ["marketing", "Branding", "Advertising"],
                "engage potential customers": ["AskReddit", "smallbusiness", "Entrepreneur"],
                "grow web traffic": ["SEO", "DigitalMarketing", "growthhacking"]
            }
            target_subreddits = set()
            # for goal in state["goals"]:
            #     target_subreddits.update(goal_subreddits.get(goal.lower(), []))
            
            keywords = []  # define keywords regardless of condition
            if not state["restrict_to_goal_subreddits"]:
                keywords = state["company_keywords"]
                if state["retries"] > 0:
                    keywords = [f"{kw} {state['retries']}" for kw in keywords]
            async def search_keyword(kw):
                result = set()
                subreddits = reddit.subreddits.search(kw, limit=10)
                async for subreddit in subreddits:
                    result.add(subreddit.display_name)
                return result
            results = await asyncio.gather(*(search_keyword(kw) for kw in keywords))
            for res in results:
                target_subreddits.update(res)
            state["subreddits"] = list(target_subreddits)
            logger.info("Subreddits identified", agent_name=state["agent_name"], subreddits=state["subreddits"])
    except Exception as e:
        state["error"] = f"Subreddit search failed: {str(e)}"
        logger.error("Subreddit search failed", agent_name=state["agent_name"], error=str(e))
    return state
        

@retry(stop = stop_after_attempt(3), wait=wait_fixed(2))
async def fetch_posts_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    try:
        async with await get_reddit_client() as reddit:
            posts = []
            cutoff_time = datetime.utcnow() - timedelta(days=state["max_age_days"])
            async def fetch_subreddit(subreddit_name: str):
                subreddit_posts = []
                try:
                    subreddit = await reddit.subreddit(subreddit_name)
                    async for submission in subreddit.hot(limit=100):
                        if submission.score < state["min_upvotes"] or datetime.utcfromtimestamp(submission.created_utc) < cutoff_time:
                            continue

                        subreddit_posts.append({
                            "subreddit": subreddit_name,
                            "post_id": submission.id,
                            "post_title": submission.title,
                            "post_body": submission.selftext,
                            "post_url": f"https://www.reddit.com{submission.permalink}"
                        })
                except asyncprawcore.exceptions.Forbidden:
                    logger.warning("Subreddit is forbidden (403)", subreddit=subreddit_name)
                except Exception as e:
                    logger.warning("Failed to fetch posts from subreddit", subreddit=subreddit_name, error=str(e))
                return subreddit_posts
            results = await asyncio.gather(*(fetch_subreddit(name) for name in state["subreddits"]), return_exceptions=False)
            for subreddit_posts in results:
                posts.extend(subreddit_posts)
            state["posts"] = posts
            print(posts)
            logger.info("Posts fetched", agent_name=state["agent_name"], post_count=len(posts))
    except Exception as e:
        state["error"] = f"Post fetching failed: {str(e)}"
        logger.error("Post fetching failed", agent_name=state["agent_name"], error=str(e))
    return state

def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_input", validate_input_node)
    graph.add_node("search_subreddits", search_subreddits_node)
    graph.add_node("fetch_posts", fetch_posts_node)
    graph.add_node("score_posts", score_posts_node)

    graph.set_entry_point("validate_input")
    
    graph.add_edge("validate_input", "search_subreddits")
    graph.add_edge("search_subreddits", "fetch_posts")
    graph.add_edge("fetch_posts", "score_posts")

    graph.add_edge("score_posts", END)
    return graph.compile()

reddit_graph = create_graph()

@router.post("/reddit/reddit-agent", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def run_reddit_agent(input: RedditAgentInput, db: AsyncSession = Depends(get_db)):
    try:
        initial_state = AgentState(
            agent_name=input.agent_name,
            goals=input.goals,
            instructions=input.instructions,
            description=input.description,
            target_audience=input.target_audience,
            company_keywords=input.company_keywords,
            min_upvotes=input.min_upvotes,
            max_age_days=input.max_age_days,
            restrict_to_goal_subreddits=input.restrict_to_goal_subreddits,
            subreddits=[],
            posts=[],
            retries=0,
            error=None,
            db=db  # Pass db session to state
        )
        
        result = await reddit_graph.ainvoke(initial_state)
        
        

    except Exception as e:
        logger.error("Reddit agent processing failed", agent_name=input.agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
