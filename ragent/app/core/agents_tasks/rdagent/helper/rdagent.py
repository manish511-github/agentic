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
from app.models import WebsiteDataModel, RedditPostModel
from app.database import get_db
from functools import lru_cache

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
    target_audience:str
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


async def get_reddit_client():
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD
    )
    return reddit

GOAL_MAPPING = {
    "lead_generation": "grow web traffic",
    "brand_awareness": "increase brand awareness",
    "engagement": "engage potential customers",
    "support": "engage potential customers"  # Mapping support to engagement as it's closest
}

def map_agent_goals(agent_goals: List[str]) -> List[str]:
    """Map agent goals to Reddit agent goals"""
    mapped_goals = []
    for goal in agent_goals:
        mapped_goal = GOAL_MAPPING.get(goal.lower())
        if mapped_goal and mapped_goal not in mapped_goals:
            mapped_goals.append(mapped_goal)
    return mapped_goals if mapped_goals else ["increase brand awareness"]

async def validate_input_node(state: AgentState) -> AgentState:
    valid_goals = ["increase brand awareness", "engage potential customers", "grow web traffic"]
    if state["goals"]:
        mapped_goals = map_agent_goals(state["goals"])
        state["goals"] = mapped_goals

    if not all(goal.lower() in valid_goals for goal in state["goals"]):
        state["error"] = f"Invalid goals. Choose from: {valid_goals}"
        logger.info(f"Invalid goals. Choose from: {valid_goals}")
        return state
    
    state["retries"] = 0
    state["subreddits"] = []
    state["posts"] = []
    logger.info("Input validated", agent_name=state["agent_name"])
    return state

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
async def generate_keywords(keywords: List[str], expectation: str) -> List[str]:
    cache_key = f"keywords:{hashlib.sha256((','.join(keywords) + expectation).encode()).hexdigest()}"
    # cached_result = await redis_client.get(cache_key)
    # if cached_result:
    #     logger.info("Cache hit for keyword expansion")
    #     return json.loads(cached_result)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY)
        keyword_prompt = PromptTemplate(
            input_variables=["keywords", "expectation"],
            template="Generate 5-10 related keywords for '{keywords}' that align with the expectation '{expectation}'. Return a valid JSON list without any markdown formatting or code blocks."
        )
        keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt)

        raw_output = await keyword_chain.arun(keywords=", ".join(keywords), expectation=expectation)
        logger.warning("Raw LLM output: %r", raw_output)
        
        try:
            expanded = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("LLM returned invalid JSON: %s", raw_output)
            raise e
        result = list(set(keywords + expanded))[:15]
        logger.info("Generated keywords", keywords_count=len(result), keywords=result[:5])  # Only log first 5 keywords
        # await redis_client.setex(cache_key, 3600, json.dumps(result))
        return result
    except Exception as e:
        logger.warning("Keyword expansion failed", error=str(e))
        return keywords

async def search_subreddits_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    
    try:
        async with await get_reddit_client() as reddit:
            # Configuration
            MAX_SUBREDDITS_TO_CONSIDER = 50
            FINAL_SUBREDDIT_LIMIT = 5
            REDDIT_BATCH_SIZE = 10
            LLM_BATCH_SIZE = 10  # Smaller batches for LLM processing
            LLM_RATE_LIMIT_DELAY = 1  # Seconds between LLM calls
            
            goal_subreddits = {
                "increase brand awareness": ["marketing", "Branding", "Advertising"],
                "engage potential customers": ["AskReddit", "smallbusiness", "Entrepreneur"],
                "grow web traffic": ["SEO", "DigitalMarketing", "growthhacking"]
            }
            
            # Use input keywords
            keywords = state["company_keywords"][:8]
            if state["retries"] > 0:
                keywords = [f"{kw} {state['retries']}" for kw in keywords]
            logger.info("Using keywords", agent_name=state["agent_name"], keywords=keywords)

            # Generate cache key
            query = f"{state['expectation']} Description: {state['description']} Target audience: {state['target_audience']}"
            cache_key = f"subreddits_data:{hashlib.sha256((query + ','.join(keywords)).encode()).hexdigest()}"
            
            # Try to get cached results
            # cached_subreddits = await redis_client.get(cache_key)
            cached_subreddits = None
            subreddit_data = []
            
            if cached_subreddits:
                logger.info("Cache hit for subreddit data", agent_name=state["agent_name"])
                subreddit_data = json.loads(cached_subreddits)
            else:
                # Phase 1: Subreddit Discovery
                target_subreddits = set()
                
                if state["restrict_to_goal_subreddits"]:
                    for goal in state["goals"]:
                        target_subreddits.update(goal_subreddits.get(goal.lower(), []))
                else:
                    async def search_keyword(kw):
                        result = set()
                        try:
                            subreddits = reddit.subreddits.search(kw, limit=10)
                            async for subreddit in subreddits:
                                result.add(subreddit.display_name)
                        except Exception as e:
                            logger.warning("Keyword search failed", keyword=kw, error=str(e))
                        return result
                    
                    # Process in batches with rate limiting
                    for i in range(0, len(keywords), REDDIT_BATCH_SIZE):
                        batch = keywords[i:i+REDDIT_BATCH_SIZE]
                        results = await asyncio.gather(*(search_keyword(kw) for kw in batch))
                        for res in results:
                            target_subreddits.update(res)
                        if len(target_subreddits) >= MAX_SUBREDDITS_TO_CONSIDER:
                            break
                        await asyncio.sleep(0.5)  # Reddit API rate limiting
                
                logger.info("Initial subreddit candidates", 
                          agent_name=state["agent_name"], 
                          subreddit_data=target_subreddits)

                # Phase 2: Subreddit Data Collection
                if target_subreddits:
                    async def fetch_subreddit_data(name):
                        try:
                            subreddit = await reddit.subreddit(name)
                            await subreddit.load()
                            description = subreddit.public_description or subreddit.title or ""
                            subscribers = getattr(subreddit, 'subscribers', 0)
                            return {
                                "name": name,
                                "description": description,
                                "subscribers": subscribers,
                                "active": (subscribers > 1000)  # Basic activity indicator
                            } if description else None
                        except Exception as e:
                            logger.warning("Subreddit fetch failed", subreddit=name, error=str(e))
                            return None

                    all_subreddits = list(target_subreddits)[:MAX_SUBREDDITS_TO_CONSIDER]
                    subreddit_data = []
                    
                    for i in range(0, len(all_subreddits), REDDIT_BATCH_SIZE):
                        batch = all_subreddits[i:i+REDDIT_BATCH_SIZE]
                        tasks = [fetch_subreddit_data(name) for name in batch]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        subreddit_data.extend([
                            r for r in results 
                            if r and not isinstance(r, Exception)
                        ])
                        
                        if i + REDDIT_BATCH_SIZE < len(all_subreddits):
                            await asyncio.sleep(1)  # Reddit API rate limiting
                
                    logger.info("Subreddit data collected", 
                              agent_name=state["agent_name"], 
                              count=len(subreddit_data), subreddit_data =subreddit_data)
                    
                    # Cache the results
                    # await redis_client.setex(cache_key, 3600, json.dumps(subreddit_data))

            # Fallback if no data found
            if not subreddit_data:
                logger.warning("No subreddit data found, using fallback", agent_name=state["agent_name"])
                for goal in state["goals"]:
                    target_subreddits.update(goal_subreddits.get(goal.lower(), []))
                state["subreddits"] = list(target_subreddits)[:FINAL_SUBREDDIT_LIMIT]
                return state

            # Phase 3: Initial Filtering
            def calculate_score(data):
                score = 0
                text = (data["name"] + " " + data["description"]).lower()
                for kw in keywords:
                    if kw.lower() in text:
                        score += 1
                subscriber_factor = math.log10(data.get("subscribers", 1) + 1)
                return (score / max(1, len(keywords))) * subscriber_factor

            scored_subreddits = sorted(
                [{"data": d, "score": calculate_score(d)} for d in subreddit_data],
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Take top candidates for LLM processing
            top_candidates = [item["data"] for item in scored_subreddits[:20]]
            
            # Phase 4: Chunked LLM Processing
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=GOOGLE_API_KEY,
                max_retries=2,
                timeout=30
            )
            
            prompt = PromptTemplate(
                input_variables=["subreddits", "expectation", "description", "target_audience", "limit"],
                template=(
                    "Evaluate these subreddits for relevance to our marketing needs. "
                    "Follow these instructions carefully:\n\n"
                    "PRODUCT DETAILS:\n"
                    "- Goals: {expectation}\n"
                    "- Description: {description}\n"
                    "- Target Audience: {target_audience}\n\n"
                    "SUBREDDITS TO EVALUATE:\n"
                    "{subreddits}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Select exactly {limit} most relevant subreddits\n"
                    "2. Consider relevance, activity level, and audience match\n"
                    "3. Return ONLY a valid JSON array formatted exactly like this example:\n"
                    "   [\"subreddit1\", \"subreddit2\", \"subreddit3\"]\n"
                    "4. DO NOT include any backticks, markdown formatting, or extra explanation\n"
                    "5. DO NOT wrap the output in a code block\n"
                    "6. Ensure all subreddit names are from the provided list and spelled exactly\n\n"
                    "OUTPUT REQUIREMENTS:\n"
                    "- Must be a valid JSON array of strings\n"
                    "- Must contain exactly {limit} items\n"
                    "- Must use double quotes for each string\n"
                    "- Must NOT include backticks, markdown, or any extra text\n\n"
                    "IMPORTANT:\n"
                    "- NO markdown formatting (no ```json ... ```)\n"
                    "- NO text before or after the JSON array\n\n"
                    "YOUR RESPONSE:"
                )
            )
            
            final_subreddits = []
            
            # Process candidates in smaller batches for LLM
            for i in range(0, len(top_candidates), LLM_BATCH_SIZE):
                batch = top_candidates[i:i+LLM_BATCH_SIZE]
                
                # Create a unique cache key for this batch
                batch_cache_key = f"subreddits_llm:{hashlib.sha256((query + ','.join([d['name'] for d in batch])).encode()).hexdigest()}"
                # cached_batch = await redis_client.get(batch_cache_key)
                cached_batch = None
                
                if cached_batch:
                    logger.info("LLM batch cache hit", agent_name=state["agent_name"])
                    batch_results = json.loads(cached_batch)
                else:
                    try:
                        chain = LLMChain(llm=llm, prompt=prompt)
                        subreddits_input = "\n".join(
                            f"{d['name']} (Subscribers: {d.get('subscribers', 'N/A')}): {d['description']}"
                            for d in batch
                        )
                        
                        result = await chain.arun(
                            subreddits=subreddits_input,
                            expectation=state["expectation"],
                            description=state["description"],
                            target_audience=state["target_audience"],
                            limit=min(LLM_BATCH_SIZE, FINAL_SUBREDDIT_LIMIT - len(final_subreddits))
                        )
                        batch_results = json.loads(result)
                        logger.info("LLM batch results", results_count=len(batch_results), results=batch_results[:3])  # Only log first 3 results
                        # await redis_client.setex(batch_cache_key, 3600, json.dumps(batch_results))
                    except json.JSONDecodeError as e:
                        logger.warning("LLM batch failed - JSON decode error", error=str(e))
                        # Fallback to score-based selection
                        batch_results = [d["name"] for d in batch[:FINAL_SUBREDDIT_LIMIT - len(final_subreddits)]]
                    except Exception as e:
                        logger.warning("LLM batch failed", error=str(e))
                        batch_results = []
                        # Implement exponential backoff if needed
                        await asyncio.sleep(LLM_RATE_LIMIT_DELAY * 2)
                    
                    # Small delay between LLM calls to avoid rate limits
                    await asyncio.sleep(LLM_RATE_LIMIT_DELAY)
                
                # Validate batch results against our candidates
                valid_results = [
                    name for name in batch_results 
                    if name in [d["name"] for d in batch]
                ]
                final_subreddits.extend(valid_results)
                
                # Early exit if we have enough
                if len(final_subreddits) >= FINAL_SUBREDDIT_LIMIT:
                    break
            
            # Final fallback if LLM processing didn't yield enough results
            if len(final_subreddits) < FINAL_SUBREDDIT_LIMIT:
                remaining = FINAL_SUBREDDIT_LIMIT - len(final_subreddits)
                fallback = [d["name"] for d in top_candidates if d["name"] not in final_subreddits][:remaining]
                final_subreddits.extend(fallback)
            
            state["subreddits"] = final_subreddits[:FINAL_SUBREDDIT_LIMIT]
            logger.info("Final subreddits selected", 
                      agent_name=state["agent_name"], 
                      subreddits=state["subreddits"])

    except Exception as e:
        state["error"] = f"Subreddit search failed: {str(e)}"
        logger.error("Subreddit search failed", agent_name=state["agent_name"], error=str(e))
    
    return state



REDDIT_RATE_LIMIT = 60  # Reddit API calls per minute
GEMINI_RATE_LIMIT = 15  # Gemini free tier limit per minute
MAX_POSTS_PER_SUBREDDIT = 5
BATCH_SIZE = 5
DELAY_BETWEEN_BATCHES = 3


async def safe_gemini_call(llm: ChatGoogleGenerativeAI, prompt: str) -> str:
    """Make rate-limited Gemini API call."""
    try:
        response = await llm.ainvoke(prompt)
        return response.content.lower()
    except Exception as e:
        logger.warning(f"Gemini call failed: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def fetch_posts_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    start_time = datetime.utcnow()
    state["posts"] = []

    try:
        # Validate goals
        valid_goals = ["increase brand awareness", "engage potential customers", "grow web traffic"]
        if not all(g in valid_goals for g in state["goals"]):
            state["error"] = f"Invalid goals: {[g for g in state['goals'] if g not in valid_goals]}"
            logger.error("Invalid goals", agent_name=state["agent_name"], invalid_goals=[g for g in state["goals"] if g not in valid_goals])
            return state

        # Initialize LLM with error handling
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=1,
            temperature=0.2
        )

        async with await get_reddit_client() as reddit:
            cutoff_time = datetime.utcnow() - timedelta(days=state["max_age_days"])
            BATCH_SIZE = 3  # Process 3 subreddits at a time
            MAX_POSTS_PER_SUBREDDIT = 5  # Focus on quality
            DELAY_BETWEEN_BATCHES = 1.0

            async def process_subreddit(subreddit_name: str) -> List[Dict]:
                """Process subreddit with LLM-based relevance scoring."""
                posts = []
                try:
                    subreddit = await reddit.subreddit(subreddit_name)
                    listings = [
                        ('hot', subreddit.hot(limit=25)),
                        ('new', subreddit.new(limit=25)),
                        ('top', subreddit.top('week', limit=25))
                    ]
                    logger.debug("Processing listings", subreddit=subreddit_name, listing_count=len(listings))
                    # Collect candidate posts
                    candidates = []
                    for sort_method, listing in listings:
                        async for submission in listing:
                            candidates.append({
                                "submission": submission,
                                "sort_method": sort_method,
                                "title": submission.title,
                                "selftext": submission.selftext
                            })
                    if not candidates:
                        return posts

                    # Cache key for LLM results
                    cache_key = f"posts:{subreddit_name}:{hashlib.sha256((state['expectation'] + ','.join(state['goals']) + state['description'] + ','.join([c['title'] + c['selftext'] for c in candidates])).encode()).hexdigest()}"
                    # cached_posts = await redis_client.get(cache_key)
                    cached_posts =None
                    if cached_posts:
                        logger.info("Cache hit for posts", subreddit=subreddit_name)
                        posts = json.loads(cached_posts)
                    else:
                        # LLM prompt for batch scoring
                        prompt = (
                            f"Score the relevance (0-1, float) of each post for a product with the following details:\n"
                            f"Product Description: {state['description']}\n"
                            f"Goals: {', '.join(state['goals'])} (ensure posts support these marketing objectives)\n"
                            f"Expectation: {state['expectation']} (posts should match this content focus)\n"
                            f"Target Audience: {state['target_audience']}\n"
                            f"Keywords: {', '.join(state['company_keywords'])}\n\n"
                            "Posts:\n" +
                            "\n".join([
                                f"Post {i}: {c['title']}\n{c['selftext'][:500]}"
                                for i, c in enumerate(candidates, 1)
                            ]) +
                            f"\n\nReturn ONLY a JSON-formatted list of relevance scores (0-1 floats), "
                            f"one for each post listed above. You must return exactly {len(candidates)} scores in the same order as the posts. "
                            "Do not include any explanations, extra text, or Markdown. Just return the list.\n"
                            "Example:\n[0.7, 0.3, 0.9, 0.5]"
                        )

                        try:
                            response = await safe_gemini_call(llm, prompt)
                            cleaned_response = re.sub(r'^```json\n|\n```$', '', response.strip())
                            scores = json.loads(cleaned_response)

                            # if len(scores) != len(candidates):
                            #     logger.info(f"Score length: {len(scores)}, Candidate length: {len(candidates)}")
                            #     raise ValueError("Mismatch in scores length")
                            
                            # Filter posts with score >= 0.75
                            for candidate, score in zip(candidates, scores):
                                if isinstance(score, (int, float)) and score >= 0.3:
                                    submission = candidate["submission"]
                                    posts.append({
                                        "subreddit": subreddit_name,
                                        "post_id": submission.id,
                                        "post_title": submission.title,
                                        "post_body": submission.selftext,
                                        "post_url": f"https://www.reddit.com{submission.permalink}",
                                        "relevance_score": score,
                                        "comment_count": submission.num_comments,
                                        "upvote_comment_ratio": submission.score / max(1, submission.num_comments),
                                        "upvotes": submission.score,
                                        "created": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                                        "sort_method": candidate["sort_method"]
                                    })
                                    

                        except Exception as e:
                            logger.warning(f"LLM scoring failed for {subreddit_name}: {str(e)}")
                            # Fallback: Keyword matching
                            for candidate in candidates:
                                content = (candidate["title"] + " " + candidate["selftext"]).lower()
                                score = sum(1 for kw in state["company_keywords"] if kw.lower() in content) / max(1, len(state["company_keywords"]))
                                if score >= 0.75:
                                    submission = candidate["submission"]
                                    posts.append({
                                        "subreddit": subreddit_name,
                                        "post_id": submission.id,
                                        "post_title": submission.title,
                                        "post_body": submission.selftext,
                                        "post_url": f"https://www.reddit.com{submission.permalink}",
                                        "relevance_score": score,
                                        "hybrid_score": score,
                                        "comment_count": submission.num_comments,
                                        "upvote_comment_ratio": submission.score / max(1, submission.num_comments),
                                        "upvotes": submission.score,
                                        "created": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                                        "sort_method": candidate["sort_method"]
                                    })

                        # Limit to MAX_POSTS_PER_SUBREDDIT
                        # posts = sorted(posts, key=lambda x: x["relevance_score"], reverse=True)[:MAX_POSTS_PER_SUBREDDIT]
                        # await redis_client.setex(cache_key, 3600, json.dumps(posts))

                    return posts

                except Exception as e:
                    logger.warning(f"Error processing {subreddit_name}: {str(e)}")
                    return posts

            # Process subreddits in batches
            for i in range(0, len(state["subreddits"]), BATCH_SIZE):
                batch = state["subreddits"][i:i + BATCH_SIZE]
                results = await asyncio.gather(
                    *(process_subreddit(name) for name in batch),
                    return_exceptions=True
                )

                for result in results:
                    if isinstance(result, list):
                        state["posts"].extend(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Batch processing error: {str(result)}")

                if i + BATCH_SIZE < len(state["subreddits"]):
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)

            # Sort and limit final results
            state["posts"] = sorted(
                state["posts"],
                key=lambda x: (x["relevance_score"], x["upvotes"], x["created"]),
                reverse=True
            )
            logger.info(
                "Posts fetched successfully",
                agent_name=state["agent_name"],
                post_count=len(state["posts"]),
                posts=state["posts"],
                duration_sec=(datetime.utcnow() - start_time).total_seconds()
            )

    except Exception as e:
        state["error"] = f"Post fetching failed: {str(e)}"
        logger.error(
            "Fetch failure",
            error=str(e),
            stack_info=True,
            agent_name=state.get("agent_name", "unknown")
        )

    return state

def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_input", validate_input_node)
    graph.add_node("search_subreddits", search_subreddits_node)
    graph.add_node("fetch_posts", fetch_posts_node)

    graph.set_entry_point("validate_input")
    
    graph.add_edge("validate_input", "search_subreddits")
    graph.add_edge("search_subreddits", "fetch_posts")
    graph.add_edge("fetch_posts", END)
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
            db=db  # Pass db session to state
        )
        
        result = await reddit_graph.ainvoke(initial_state)
        
        

    except Exception as e:
        logger.error("Reddit agent processing failed", agent_name=input.agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")