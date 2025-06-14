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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
from qdrant_client import QdrantClient
from qdrant_client.http import models
# Comment out SentenceTransformer but keep for reference
# from sentence_transformers import SentenceTransformer

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
    llm: Optional[ChatGoogleGenerativeAI]  # Added to store LLM instance


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
        logger.info(result)
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
            # query = f"{state['expectation']} Description: {state['description']} Target audience: {state['target_audience']}"
            # cache_key = f"subreddits_data:{hashlib.sha256((query + ','.join(keywords)).encode()).hexdigest()}"
            
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
                        await asyncio.sleep(1)  # Reddit API rate limiting
                
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
                                "active": (subscribers > 50)  # Basic activity indicator
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
                model="gemini-1.5-flash",
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
                # batch_cache_key = f"subreddits_llm:{hashlib.sha256((query + ','.join([d['name'] for d in batch])).encode()).hexdigest()}"
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
                        logger.info(batch_results)
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
        logger.warning("Skipping fetch_posts_node due to existing error", 
                      agent_name=state.get("agent_name", "unknown"),
                      error=state.get("error"))
        return state

    start_time = datetime.utcnow()
    state["posts"] = []

    try:
        # Validate goals
        valid_goals = ["increase brand awareness", "engage potential customers", "grow web traffic"]
        if not all(g in valid_goals for g in state["goals"]):
            state["error"] = f"Invalid goals: {[g for g in state['goals'] if g not in valid_goals]}"
            logger.error("Invalid goals", 
                        agent_name=state["agent_name"], 
                        invalid_goals=[g for g in state["goals"] if g not in valid_goals],
                        valid_goals=valid_goals)
            return state

        logger.info("Starting fetch_posts_node", 
                   agent_name=state["agent_name"],
                   goals=state["goals"],
                   keywords=state["company_keywords"],
                   subreddits=state["subreddits"])

        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        embedding_model = get_embedding_model()
        logger.info("Initialized Qdrant client and embedding model")

        # Create Qdrant collection
        collection_name = "subreddit_posts"
        try:
            # Try to delete existing collection
            try:
                qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection {collection_name}")
            except Exception as e:
                if "not found" not in str(e).lower():
                    logger.error(f"Error deleting collection: {str(e)}")
                    raise

            # Create new collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Size of Gemini embedding model
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection {collection_name}")
        except Exception as e:
            logger.error(f"Error managing collection: {str(e)}")
            raise

        async with await get_reddit_client() as reddit:
            BATCH_SIZE = 50  # Process 50 posts at a time
            current_batch = []
            posts = []
            seen_post_ids = set()
            total_posts_processed = 0
            total_keyword_matches = 0

            async def process_subreddit(subreddit_name: str) -> List[Dict]:
                """Process subreddit by searching for keywords and storing in collection."""
                subreddit_posts = []
                subreddit_start_time = datetime.utcnow()
                subreddit_keyword_matches = 0
                try:
                    logger.info(f"Processing subreddit: {subreddit_name}")
                    
                    # Get subreddit with error handling
                    try:
                        subreddit = await reddit.subreddit(subreddit_name)
                        await subreddit.load()  # Explicitly load subreddit data
                        logger.info(f"Successfully loaded subreddit: {subreddit_name}")
                    except Exception as e:
                        logger.error(f"Failed to load subreddit {subreddit_name}: {str(e)}")
                        return subreddit_posts
                    
                    # Create search query from keywords - use OR for each keyword
                    search_queries = []
                    for keyword in state["company_keywords"]:
                        # Add exact match
                        search_queries.append(f'"{keyword}"')
                        # Add partial match
                        search_queries.append(keyword)
                    
                    logger.info(f"Generated search queries for {subreddit_name}: {search_queries}")
                    
                    # Try each search query
                    for search_query in search_queries:
                        try:
                            logger.info(f"Searching {subreddit_name} with query: {search_query}")
                            # Search in subreddit with keywords - try without time filter first
                            async for submission in subreddit.search(search_query, sort="relevance", time_filter="month", limit=25):
                                if submission.id in seen_post_ids:
                                    continue
                                seen_post_ids.add(submission.id)

                                # Calculate keyword-based relevance score
                                content = (submission.title + " " + submission.selftext).lower()
                                keyword_matches = sum(1 for kw in state["company_keywords"] if kw.lower() in content)
                                keyword_relevance = min(1.0, keyword_matches / max(1, len(state["company_keywords"])))

                                # Create post data
                                post_data = {
                                    "subreddit": subreddit_name,
                                    "post_id": submission.id,
                                    "post_title": submission.title,
                                    "post_body": submission.selftext,
                                    "post_url": f"https://www.reddit.com{submission.permalink}",
                                    "upvotes": submission.score,
                                    "comment_count": submission.num_comments,
                                    "created": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                                    "keyword_relevance": keyword_relevance,
                                    "text": f"{submission.title} {submission.selftext}"
                                }
                                subreddit_posts.append(post_data)
                                subreddit_keyword_matches += keyword_matches
                                logger.info(f"Found post in {subreddit_name}: {submission.title}")
                            
                            # Add a 3-second delay between searches to respect rate limits
                            await asyncio.sleep(3)  # 3 seconds delay between searches
                            
                        except Exception as e:
                            logger.warning(f"Search failed for query '{search_query}' in {subreddit_name}: {str(e)}")
                            continue

                    logger.info(f"Completed processing {subreddit_name}", 
                              posts_found=len(subreddit_posts),
                              keyword_matches=subreddit_keyword_matches,
                              duration_sec=(datetime.utcnow() - subreddit_start_time).total_seconds())
                except Exception as e:
                    logger.error(f"Error processing {subreddit_name}: {str(e)}", 
                               subreddit=subreddit_name,
                               error=str(e))
                return subreddit_posts

            # Process subreddits in batches
            for i in range(0, len(state["subreddits"]), BATCH_SIZE):
                batch = state["subreddits"][i:i + BATCH_SIZE]
                batch_start_time = datetime.utcnow()
                logger.info(f"Processing batch {i//BATCH_SIZE + 1}", 
                          batch_size=len(batch),
                          subreddits=batch)

                try:
                    results = await asyncio.gather(
                        *(process_subreddit(name) for name in batch),
                        return_exceptions=True
                    )

                    for result in results:
                        if isinstance(result, list):
                            posts.extend(result)
                            total_posts_processed += len(result)
                        elif isinstance(result, Exception):
                            logger.error(f"Batch processing error: {str(result)}")

                    logger.info(f"Completed batch {i//BATCH_SIZE + 1}",
                              duration_sec=(datetime.utcnow() - batch_start_time).total_seconds(),
                              posts_in_batch=len(posts) - total_posts_processed)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue

            logger.info("Starting embedding generation and storage",
                      total_posts=len(posts),
                      batch_size=BATCH_SIZE)

            # Generate embeddings and store in Qdrant
            for i in range(0, len(posts), BATCH_SIZE):
                batch_posts = posts[i:i + BATCH_SIZE]
                batch_start_time = datetime.utcnow()
                texts = [post["text"] for post in batch_posts]
                
                try:
                    # Generate embeddings
                    logger.debug(f"Generating embeddings for batch {i//BATCH_SIZE + 1}")
                    embeddings = await embedding_model.aembed_documents(texts)
                    
                    # Create Qdrant points
                    points = []
                    for post, embedding in zip(batch_posts, embeddings):
                        points.append(models.PointStruct(
                            id=abs(hash(post["post_id"])) % (2**63),
                            vector=embedding,
                            payload={
                                "submission_id": post["post_id"],
                                "title": post["post_title"],
                                "content": post["post_body"],
                                "url": post["post_url"],
                                "created_utc": datetime.fromisoformat(post["created"]).timestamp(),
                                "subreddit": post["subreddit"],
                                "score": post["upvotes"],
                                "num_comments": post["comment_count"],
                                "keyword_relevance": post["keyword_relevance"]
                            }
                        ))
                    
                    # Store in Qdrant
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    logger.info(f"Stored batch {i//BATCH_SIZE + 1} in Qdrant",
                              posts_in_batch=len(batch_posts),
                              duration_sec=(datetime.utcnow() - batch_start_time).total_seconds())
                except Exception as e:
                    logger.error(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}",
                               batch_index=i//BATCH_SIZE + 1,
                               error=str(e))

            # Perform semantic search on collected posts
            try:
                logger.info("Starting semantic search")
                # Create a comprehensive query from company data
                query_text = f"""
                {state['description']}
                {', '.join(state['goals'])}
                {', '.join(state['company_keywords'])}
                {state['target_audience']}
                {state['expectation']}
                """
                
                # Generate embedding for the query
                query_embedding = await embedding_model.aembed_query(query_text)
                
                # Search for semantically similar posts
                semantic_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=100,
                    score_threshold=0.5  # Minimum similarity threshold
                )

                # Update posts with semantic relevance scores
                post_dict = {post["post_id"]: post for post in posts}
                semantic_posts = []
                for result in semantic_results:
                    post_id = result.payload["submission_id"]
                    if post_id in post_dict:
                        post = post_dict[post_id]
                        # Add semantic score
                        post["semantic_relevance"] = result.score
                        # Calculate combined relevance
                        post["combined_relevance"] = (
                            0.8 * result.score + 
                            0.2 * post["keyword_relevance"]
                        )
                        semantic_posts.append(post)

                # Sort by combined relevance
                semantic_posts.sort(key=lambda x: x["combined_relevance"], reverse=True)
                state["posts"] = semantic_posts

                logger.info("Semantic search completed",
                          total_posts=len(semantic_posts),
                          avg_semantic_score=sum(p["semantic_relevance"] for p in semantic_posts) / len(semantic_posts) if semantic_posts else 0)

            except Exception as e:
                logger.error(f"Semantic search failed: {str(e)}",
                           error=str(e))
                # Fallback to keyword-based sorting if semantic search fails
                state["posts"] = sorted(
                    posts,
                    key=lambda x: (x["keyword_relevance"], x["upvotes"]),
                    reverse=True
                )
                logger.info("Falling back to keyword-based sorting",
                          total_posts=len(state["posts"]))

            logger.info(
                "Posts fetched and analyzed successfully",
                agent_name=state["agent_name"],
                post_count=len(state["posts"]),
                total_posts_processed=total_posts_processed,
                total_keyword_matches=total_keyword_matches,
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

# Comment out model cache directory setup (was used for SentenceTransformer)
# Set up model cache directory
# MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")
# os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Initialize global variables for caching
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Initializing Google Gemini embedding model")
        # Using Gemini embedding model
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        # Commented out SentenceTransformer implementation
        # _embedding_model = SentenceTransformer(
        #     'all-MiniLM-L6-v2',
        #     cache_folder=MODEL_CACHE_DIR
        # )
    return _embedding_model

async def parallel_searches(state: AgentState):
    direct_search = search_posts_directly_node(state)
    targeted_search = fetch_posts_node(state)
    direct_results, targeted_results = await asyncio.gather(direct_search, targeted_search, return_exceptions=True) #Prevent one failure from breaking flow

    #Store results 
    state["direct_posts"] = direct_results.get("posts", []) if not isinstance(direct_results, Exception) else []
    state["targeted_posts"] = targeted_results.get("posts", []) if not isinstance(targeted_results, Exception) else []

    return state


async def search_posts_directly_node(state: AgentState) -> AgentState:
    """Search Reddit posts directly using the Reddit API search functionality."""
    if state.get("error"):
        return state

    try:
        # Get Reddit client
        reddit = await get_reddit_client()
        
        # Initialize LLM if not already in state
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY)
        
        # Initialize Qdrant client and get cached embedding model
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        embedding_model = get_embedding_model()  # Use cached model
        
        # Create Qdrant collection (delete if exists and create new)
        collection_name = "reddit_posts"
        try:
            # Try to delete existing collection
            try:
                qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection {collection_name}")
            except Exception as e:
                if "not found" not in str(e).lower():
                    logger.error(f"Error deleting collection: {str(e)}")
                    raise

            # Create new collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Size of Gemini embedding-001 model
                    # size=384,  # Size of all-MiniLM-L6-v2 embeddings
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error managing collection: {str(e)}")
            raise

        # Prepare company data for query generation
        company_data = {
            "agent_name": state.get("agent_name", ""),
            "goals": state.get("goals", []),
            "instructions": state.get("instructions", ""),
            "company_keywords": state.get("company_keywords", []),
            "description": state.get("description", ""),
            "target_audience": state.get("target_audience", ""),
            "expectation": state.get("expectation", "")
        }

        # Generate search queries using Gemini
        prompt = f"""Based on the following company information, generate a comprehensive list of search queries for finding relevant Reddit discussions. 
        Focus on generating queries that will help find posts about our product, its broader category, target audience, and industry.
        Keep each query under 50 characters and make them specific and relevant.

        Company Information:
        - Name: {company_data['agent_name']}
        - Goals: {', '.join(company_data['goals'])}
        - Keywords: {', '.join(company_data['company_keywords'])}
        - Description: {company_data['description']}
        - Target Audience: {company_data['target_audience']}
        - Expected Content: {company_data['expectation']}

        Generate queries in the following categories:
        1. Core product terms (specific to our product)
        2. Product category terms (broader category our product belongs to)
        3. Feature-specific terms (both our product and similar products)
        4. Platform-specific terms
        5. Use case terms (both specific and general use cases)
        6. Target audience terms (including broader audience segments)
        7. Industry terms (both specific and general industry discussions)
        8. Alternative solutions (competitors and similar products)
        9. Problem space terms (common issues our product solves)
        10. Market trends and discussions

        For each category, include:
        - Specific queries about our product
        - Broader queries about the product category
        - Related industry discussions
        - Common pain points and solutions
        - Market trends and developments

        OUTPUT REQUIREMENTS:
        - Must be a valid JSON array of strings
        - Each string must be a search query
        - Each query must be under 50 characters
        - Must use double quotes for strings
        - Must NOT include backticks, markdown, or any extra text

        IMPORTANT:
        - NO markdown formatting (no ```json ... ```)
        - NO text before or after the JSON array
        - NO explanations or additional text

        YOUR RESPONSE:"""

        # Get Gemini response using safe_gemini_call
        gemini_response = await safe_gemini_call(llm, prompt)
        
        try:
            # Clean the response by removing markdown formatting
            cleaned_response = gemini_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            # Parse the cleaned response as JSON
            search_queries = json.loads(cleaned_response)
            if not isinstance(search_queries, list):
                raise ValueError("Invalid response format")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse Gemini response", 
                        error=str(e),
                        response=gemini_response)
            # Fallback to basic queries if parsing fails
            search_queries = company_data['company_keywords']
        
        # Remove duplicates and ensure queries are short enough
        search_queries = list(set(search_queries + company_data['company_keywords']))
        search_queries = [q for q in search_queries if len(q) <= 50]  # Keep queries short
        
        logger.info("Generated search queries using Gemini", 
                   agent_name=state["agent_name"],
                   queries=search_queries)

        # Initialize batch processing variables
        BATCH_SIZE = 50  # Process 50 posts at a time
        EMBEDDING_BATCH_SIZE = 10  # Process 10 embeddings at a time
        current_batch = []
        posts = []
        seen_post_ids = set()

        async def process_submission_batch(submissions, current_query):
            """Process a batch of submissions in parallel."""
            tasks = []
            for submission in submissions:
                if submission.id in seen_post_ids:
                    continue
                seen_post_ids.add(submission.id)
                
                # Calculate keyword-based relevance score
                content = (submission.title + " " + submission.selftext).lower()
                keyword_matches = sum(1 for kw in search_queries if kw.lower() in content)
                keyword_relevance = min(1.0, keyword_matches / max(1, len(state["company_keywords"])))
                
                # Create post data
                post_data = {
                    "subreddit": submission.subreddit.display_name,
                    "post_id": submission.id,
                    "post_title": submission.title,
                    "post_body": submission.selftext,
                    "post_url": f"https://www.reddit.com{submission.permalink}",
                    "upvotes": submission.score,
                    "comment_count": submission.num_comments,
                    "created": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                    "keyword_relevance": keyword_relevance,
                    "matched_query": current_query,
                    "text": f"{submission.title} {submission.selftext}"
                }
                tasks.append(post_data)
            return tasks

        async def generate_embeddings_batch(texts):
            """Generate embeddings for a batch of texts."""
            try:
                # Using Gemini async embedding with batch processing
                embeddings = await embedding_model.aembed_documents(texts)
                return embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                return [None] * len(texts)

        async with reddit as reddit_client:
            all_subreddit = await reddit_client.subreddit("all")
            
            # Process queries in parallel
            async def process_query(query):
                submissions = []
                try:
                    async for submission in all_subreddit.search(
                        query=query,
                        sort="relevance",
                        time_filter="month",
                        limit=30
                    ):
                        submissions.append(submission)
                        if len(submissions) >= 30:
                            break
                except Exception as e:
                    logger.warning(f"Search failed for query: {query}", error=str(e))
                return submissions, query  # Return both submissions and the query

            # Process all queries in parallel
            query_tasks = [process_query(query) for query in search_queries]
            query_results = await asyncio.gather(*query_tasks)
            
            # Process submissions in batches
            for submissions, current_query in query_results:
                if not submissions:
                    continue
                    
                batch_posts = await process_submission_batch(submissions, current_query)
                
                if not batch_posts:
                    continue

                # Generate embeddings in smaller batches
                for j in range(0, len(batch_posts), EMBEDDING_BATCH_SIZE):
                    embedding_batch = batch_posts[j:j + EMBEDDING_BATCH_SIZE]
                    texts = [post["text"] for post in embedding_batch]
                    embeddings = await generate_embeddings_batch(texts)

                    # Create Qdrant points
                    for post, embedding in zip(embedding_batch, embeddings):
                        if embedding is None:
                            continue

                        current_batch.append(models.PointStruct(
                            id=abs(hash(post["post_id"])) % (2**63),
                            vector=embedding,
                            payload={
                                "submission_id": post["post_id"],
                                "title": post["post_title"],
                                "content": post["post_body"],
                                "url": post["post_url"],
                                "created_utc": datetime.fromisoformat(post["created"]).timestamp(),
                                "subreddit": post["subreddit"],
                                "score": post["upvotes"],
                                "num_comments": post["comment_count"],
                                "keyword_relevance": post["keyword_relevance"]
                            }
                        ))

                        # Add to posts list without the text field
                        post.pop("text", None)
                        posts.append(post)

                    # Process batch if it reaches the batch size
                    if len(current_batch) >= BATCH_SIZE:
                        try:
                            qdrant_client.upsert(
                                collection_name=collection_name,
                                points=current_batch
                            )
                            current_batch = []  # Clear the batch
                        except Exception as e:
                            logger.error(f"Error upserting batch: {str(e)}")
                            current_batch = []

                # Break if we have enough posts
                if len(posts) >= 500:
                    break

            # Process any remaining posts in the last batch
            if current_batch:
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=current_batch
                    )
                except Exception as e:
                    logger.error(f"Error upserting final batch: {str(e)}")

        # Perform semantic search on collected posts
        try:
            # Create a comprehensive query from company data
            query_text = f"""
            {company_data['description']}
            {', '.join(company_data['goals'])}
            {', '.join(search_queries)}
            {company_data['target_audience']}
            {company_data['expectation']}
            """
            
            # Generate embedding for the query
            # Using Gemini async embedding
            query_embedding = await embedding_model.aembed_query(query_text)
            # Commented out SentenceTransformer embedding
            # query_embedding = embedding_model.encode(query_text).tolist()
            
            # Search for semantically similar posts
            semantic_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=100,
                score_threshold=0.5  # Minimum similarity threshold
            )
            # Update posts with semantic relevance scores
            post_dict = {post["post_id"]: post for post in posts}
            semantic_posts = []
            for result in semantic_results:
                post_id = result.payload["submission_id"]
                if post_id in post_dict:
                    post = post_dict[post_id]
                    # Add semantic score
                    post["semantic_relevance"] = result.score
                    # Keep combined relevance
                    post["combined_relevance"] = (
                        0.8 * result.score + 
                        0.2 * post["keyword_relevance"]
                    )
                    semantic_posts.append(post)
            # Convert back to list and sort by combined relevance
            semantic_posts.sort(key=lambda x: x["combined_relevance"], reverse=True)
 
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            # Fallback to keyword-based sorting if semantic search fails
            posts.sort(key=lambda x: (x["keyword_relevance"], x["upvotes"]), reverse=True)
        
        # Take top 50 most relevant posts
        state["posts"] = semantic_posts
        
        logger.info("Direct post search completed with semantic ranking", 
                   agent_name=state["agent_name"],
                   posts_found=len(state["posts"]),
                   unique_queries_used=search_queries)

    except Exception as e:
        state["error"] = f"Direct post search failed: {str(e)}"
        logger.error("Direct post search failed", error=str(e))

    return state

def create_reddit_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("search_subreddits_node",search_subreddits_node)
    graph.add_node("fetch_posts_node",fetch_posts_node)
    
    # graph.add_node("parallel_searches", parallel_searches)

    # Set entry point
    graph.set_entry_point("validate_input")
    
    # Add edges
    graph.add_edge("validate_input", "search_subreddits_node")
    graph.add_edge("search_subreddits_node", "fetch_posts_node")
    graph.add_edge("fetch_posts_node", END)
    
    return graph.compile()

reddit_graph = create_reddit_graph()

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
            db=db,  # Pass db session to state
            llm=None  # Initialize llm as None
        )
        
        result = await reddit_graph.ainvoke(initial_state)
        return result
        

    except Exception as e:
        logger.error("Reddit agent processing failed", agent_name=input.agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")