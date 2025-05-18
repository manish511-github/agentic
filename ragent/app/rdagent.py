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
import spacy
import structlog
import json
import asyncpraw
import asyncprawcore  # Import asyncprawcore to handle its exceptions
from datetime import datetime, timedelta
import asyncio
from app.models import WebsiteDataModel, RedditPostModel
from app.database import get_db
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

embedding_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

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
        # for post in state["posts"]:
        #     content = f"{post['post_title']} {post['post_body']}".lower()
        #     matches = sum(1 for kw in company_keywords if kw in content)
        #     relevance_score = matches / max(len(company_keywords), 1)
        #     if relevance_score > 0.0:
        #         print(relevance_score)
        #         # print(content)
        #         post["relevance_score"] = relevance_score
        #         filtered_posts.append((post, content))
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
        # import pprint
        # pprint.pprint(filtered_posts)
        logger.info("Posts scored", agent_name=state["agent_name"], post_count=len(state["posts"]))
    except Exception as e:
        state["error"] = f"Post scoring failed: {str(e)}"
        logger.error("Post scoring failed", agent_name=state["agent_name"], error=str(e))
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
async def analyze_generative_relevance(content: str, expectation: str, keywords: List[str]) -> float:
    content = content[:2000]
    cache_key = f"generative_relevance:{hashlib.sha256((content + expectation).encode()).hexdigest()}"
    # cached_result = await redis_client.get(cache_key)
    # if cached_result:
    #     logger.info("Cache hit for generative relevance")
    #     return float(cached_result)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY)
        relevance_prompt = PromptTemplate(
            input_variables=["content", "expectation", "keywords"],
            template=(
            "On a scale from 0 to 1, where 1 is highly relevant and 0 is not relevant at all, "
            "rate how relevant the following content is to the expectation '{expectation}' and the keywords '{keywords}'.\n\n"
            "IMPORTANT: Respond with ONLY a single number between 0 and 1 (e.g., 0.75) and NOTHING ELSE - no explanations, no formatting, no text.\n\n"
            "Content:\n{content}"
            )
        )
        relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
        score = float(await relevance_chain.arun(content=content, expectation=expectation, keywords=", ".join(keywords)))
        score = max(min(score, 1.0), 0.0)
        # await redis_client.setex(cache_key, 3600, str(score))
        return score
    except Exception as e:
        logger.warning("Generative relevance analysis failed", error=str(e))
        return 0.0
    
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


from functools import lru_cache

REDDIT_RATE_LIMIT = 60  # Reddit API calls per minute
GEMINI_RATE_LIMIT = 15  # Gemini free tier limit per minute
MAX_POSTS_PER_SUBREDDIT = 5
BATCH_SIZE = 5
DELAY_BETWEEN_BATCHES = 3


@lru_cache(maxsize=32)
def generate_patterns(description: str, audience: str) -> Dict[str, re.Pattern]:
    """Dynamically generate regex patterns from product description and target audience."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(description.lower())
    
    solution_terms = set()
    feature_terms = set()
    audience_terms = set()

    # Extract solution terms and features
    for token in doc:
        if token.pos_ == "VERB" and len(token.lemma_) > 3:
            solution_terms.add(token.lemma_)
        elif token.pos_ == "NOUN" and len(token.text) > 3:
            feature_terms.add(token.text)

    for chunk in doc.noun_chunks:
        if len(chunk.text) > 5:
            feature_terms.add(chunk.text)

    # Process audience roles
    for role in audience.split(','):
        role = role.strip().lower()
        if role:
            role_doc = nlp(role)
            audience_terms.update(
                tok.lemma_ for tok in role_doc 
                if not tok.is_stop and not tok.is_punct and len(tok.text) > 2
            )

    return {
        'solution_terms': re.compile(
            r'\b(' + '|'.join(map(re.escape, solution_terms)) + r')\b',
            re.IGNORECASE
        ) if solution_terms else None,
        'feature_terms': re.compile(
            r'\b(' + '|'.join(map(re.escape, feature_terms)) + r')\b',
            re.IGNORECASE
        ) if feature_terms else None,
        'audience_terms': re.compile(
            r'\b(' + '|'.join(map(re.escape, audience_terms)) + r')\b',
            re.IGNORECASE
        ) if audience_terms else None,
        'problem_terms': re.compile(
            r'\b(struggl|challeng|difficult|issue|problem|pain|complex|'
            r'frustrat|hard to|insecure|vulnerable|breach|hack|complian|'
            r'requirement|need|looking for|searching|want|require|'
            r'improve|better|alternative|recommend|compare|evaluat)\b',
            re.IGNORECASE
        ),
        'question_terms': re.compile(
            r'^(what|how|why|where|which|who|can|does|do|is|are|will|'
            r'should|could|would|has|have)\b|\?',
            re.IGNORECASE
        )
    }

def create_post_dict(submission, subreddit_name, relevance_score, matches, sort_method) -> Dict:
    """Helper to create standardized post dictionary."""
    return {
        'subreddit': subreddit_name,
        'post_id': submission.id,
        'title': submission.title,
        'body': submission.selftext,
        'url': f"https://www.reddit.com{submission.permalink}",
        'upvotes': submission.score,
        'created': submission.created_utc,
        'relevance': relevance_score,
        'matches': matches,
        'sort': sort_method,
        'verified': relevance_score >= 0.7
    }

async def is_relevant_subreddit(reddit, subreddit_name: str, patterns: Dict) -> bool:
    """Check if subreddit is likely to contain relevant content."""
    try:
        subreddit = await reddit.subreddit(subreddit_name)
        description = (await subreddit.description()).lower()
        return any(
            pattern.search(description) 
            for pattern in patterns.values() 
            if pattern
        )
    except Exception:
        return True  # Default to processing if we can't check

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
    """Optimized Reddit post fetcher with integrated rate limiting and error handling."""
    if state.get("error"):
        return state

    start_time = datetime.utcnow()
    state["posts"] = []
    
    try:
        # Initialize components
        patterns = generate_patterns(state["description"], state["target_audience"])
        gemini_semaphore = asyncio.Semaphore(GEMINI_RATE_LIMIT)
        
        # Initialize LLM with error handling
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=1,
            temperature=0.2
        )

        async with await get_reddit_client() as reddit:
            cutoff_time = datetime.utcnow() - timedelta(days=state["max_age_days"])
            
            async def process_subreddit(subreddit_name: str) -> List[Dict]:
                """Process individual subreddit with error handling."""
                posts = []
                try:
                    if not await is_relevant_subreddit(reddit, subreddit_name, patterns):
                        return posts
                        
                    subreddit = await reddit.subreddit(subreddit_name)
                    listings = [
                        ('hot', subreddit.hot(limit=50)),
                        ('new', subreddit.new(limit=50)),
                        ('top', subreddit.top('week', limit=50))
                    ]

                    for sort_method, listing in listings:
                        async for submission in listing:
                            if (submission.score < state["min_upvotes"] or 
                                datetime.utcfromtimestamp(submission.created_utc) < cutoff_time or
                                submission.stickied):
                                continue

                            content = f"{submission.title}\n{submission.selftext}".lower()
                            matches = {
                                'solution': len(patterns['solution_terms'].findall(content)) if patterns['solution_terms'] else 0,
                                'feature': len(patterns['feature_terms'].findall(content)) if patterns['feature_terms'] else 0,
                                'audience': len(patterns['audience_terms'].findall(content)) if patterns['audience_terms'] else 0,
                                'problem': len(patterns['problem_terms'].findall(content)),
                                'is_question': bool(patterns['question_terms'].search(submission.title))
                            }

                            relevance_score = (
                                0.25 * min(1, matches['solution'] / 2) +
                                0.25 * min(1, matches['feature'] / 2) +
                                0.20 * min(1, matches['audience'] / 2) +
                                0.20 * min(1, matches['problem'] / 2) +
                                0.10 * matches['is_question']
                            )

                            if relevance_score >= 0.7:
                                try:
                                    async with gemini_semaphore:
                                        await asyncio.sleep(random.uniform(0.1, 0.5))  # Add jitter
                                        prompt = f"Verify (yes/no) if this relates to {state['company_keywords'][0]}:\nPost: {content[:500]}"
                                        response = await safe_gemini_call(llm, prompt)
                                        if response.startswith('y'):
                                            posts.append(create_post_dict(submission, subreddit_name, relevance_score, matches, sort_method))
                                except Exception:
                                    if relevance_score >= 0.9:
                                        posts.append(create_post_dict(submission, subreddit_name, relevance_score, matches, sort_method))
                            elif relevance_score >= 0.8:
                                posts.append(create_post_dict(submission, subreddit_name, relevance_score, matches, sort_method))

                            if len(posts) >= MAX_POSTS_PER_SUBREDDIT:
                                break

                except Exception as e:
                    logger.warning(f"Error processing {subreddit_name}: {str(e)}")
                    raise
                
                return posts

            # Process subreddits in optimized batches
            relevant_subreddits = [
                s for s in state["subreddits"] 
                if await is_relevant_subreddit(reddit, s, patterns)
            ]
            
            for i in range(0, len(relevant_subreddits), BATCH_SIZE):
                batch = relevant_subreddits[i:i + BATCH_SIZE]
                results = await asyncio.gather(
                    *(process_subreddit(name) for name in batch),
                    return_exceptions=True
                )
                
                for result in results:
                    if isinstance(result, list):
                        state["posts"].extend(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Batch processing error: {str(result)}")
                
                if i + BATCH_SIZE < len(relevant_subreddits):
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)

            # Sort and limit results
            state["posts"] = sorted(
                state["posts"],
                key=lambda x: (-x['relevance'], -x['matches']['audience'], -x['upvotes'], -x['created'])
            )[:30]
            print(state["posts"])
            logger.info(
                "Posts fetched successfully",
                agent_name=state["agent_name"],
                post_count=len(state["posts"]),
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
