from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from twikit import Client, TooManyRequests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from typing import List, Dict, Optional, TypedDict
import os
import random
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
import json
from datetime import datetime, timezone
import asyncio
from app.models import TwitterPostModel
from app.database import get_db
from asyncio import Semaphore, Lock
from collections import deque

# Initialize FastAPI router
router = APIRouter()

# Initialize logging
import logging
import sys
from logging.handlers import RotatingFileHandler

# Standard logging configuration
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
    handlers=[
        RotatingFileHandler(
            'twitter_agent.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

# Structlog configuration
structlog.configure(
    processors=[
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
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
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_EMAIL = os.getenv("TWITTER_EMAIL")
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD")

# Constants for rate limiting
MIN_WAIT_TIME = 60  # Increased to 60 seconds between requests
MAX_WAIT_TIME = 120  # Increased to 120 seconds between requests
MIN_TWEETS_PER_HASHTAG = 10
MAX_CONCURRENT_REQUESTS = 1  # Only one request at a time
RATE_LIMIT_WINDOW = 900  # Increased to 15 minutes
RATE_LIMIT_MAX_REQUESTS = 30  # Reduced to 30 requests per window
HELP_CENTER_WINDOW = 300  # 5 minutes for help-center endpoint
HELP_CENTER_MAX_REQUESTS = 10  # Maximum 10 requests per 5 minutes for help-center

# Global rate limiter
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
request_lock = Lock()
request_times = deque(maxlen=RATE_LIMIT_MAX_REQUESTS)
help_center_times = deque(maxlen=HELP_CENTER_MAX_REQUESTS)
last_request_time = None

class AgentState(TypedDict):
    agent_name: str
    goals: List[str]
    instructions: str
    description: str
    expectation: str
    target_audience: str
    company_keywords: List[str]
    min_likes: int
    max_age_days: int
    hashtags: List[str]
    tweets: List[Dict]
    retries: int
    error: Optional[str]
    db: AsyncSession

class TwitterAgentInput(BaseModel):
    agent_name: str
    goals: List[str]
    instructions: str
    description: str
    expectation: str
    target_audience: str
    company_keywords: List[str]
    min_likes: int = Field(default=10)
    max_age_days: int = Field(default=7)

class TweetQueryInput(BaseModel):
    query: str
    minimum_tweets: int = Field(default=10)
    product: str = Field(default="Top")
    output_file: Optional[str] = None

async def get_twitter_client():
    try:
        logger.info("Initializing Twitter client", username=TWITTER_USERNAME)
        client = Client('en-US')
        
        try:
            logger.info("Attempting to load existing cookies")
            await client.load_cookies('/app/cookies/twitter_cookies.json')
            logger.info("Successfully loaded existing cookies")
        except Exception as e:
            logger.warning("No existing cookies found or invalid cookies", error=str(e))
            logger.info("Attempting new login", username=TWITTER_USERNAME, email=TWITTER_EMAIL)
            
            try:
                await client.login(
                    auth_info_1=TWITTER_USERNAME,
                    auth_info_2=TWITTER_EMAIL,
                    password=TWITTER_PASSWORD,
                    cookies_file='/app/cookies/twitter_cookies.json'
                )
                logger.info("Successfully logged in and saved cookies")
            except Exception as login_error:
                logger.error("Login failed", error=str(login_error), username=TWITTER_USERNAME)
                raise HTTPException(
                    status_code=500,
                    detail=f"Twitter login failed: {str(login_error)}"
                )
        
        return client
    except Exception as e:
        logger.error("Failed to initialize Twitter client", error=str(e), stack_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Twitter client: {str(e)}"
        )

async def get_tweets(client: Client, query: str, tweets=None):
    """Get tweets with rate limiting and pagination handling"""
    try:
        if tweets is None:
            logger.info(f"Getting initial tweets for query: {query}")
            tweets = await client.search_tweet(query, 'Latest')
        else:
            wait_time = random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME)
            logger.info(f"Getting next tweets after {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            tweets = await tweets.next()
        return tweets
    except TooManyRequests as e:
        rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
        wait_time = (rate_limit_reset - datetime.now(timezone.utc)).total_seconds()
        wait_time = max(wait_time + 60, 600)  # Minimum 10 minutes wait
        logger.warning(f"Rate limit reached. Waiting until {rate_limit_reset}", 
                      wait_seconds=wait_time)
        await asyncio.sleep(wait_time)
        return await get_tweets(client, query, tweets)
    except Exception as e:
        logger.error(f"Error getting tweets: {str(e)}")
        raise

GOAL_MAPPING = {
    "lead_generation": "grow web traffic",
    "brand_awareness": "increase brand awareness",
    "engagement": "engage potential customers",
    "support": "engage potential customers"
}

def map_agent_goals(agent_goals: List[str]) -> List[str]:
    """Map agent goals to Twitter agent goals"""
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
    state["hashtags"] = []
    state["tweets"] = []
    logger.info("Input validated", agent_name=state["agent_name"])
    return state

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
async def generate_hashtags(keywords: List[str], expectation: str) -> List[str]:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY)
        hashtag_prompt = PromptTemplate(
            input_variables=["keywords", "expectation", "goals", "instructions", "description"],
            template=(
                "You are a social media marketing expert. Generate strategic hashtags for a marketing campaign.\n\n"
                "CAMPAIGN DETAILS:\n"
                "Marketing Goals: {goals}\n"
                "Campaign Instructions: {instructions}\n"
                "Product Description: {description}\n"
                "Campaign Expectation: {expectation}\n"
                "Key Keywords: {keywords}\n\n"
                "HASHTAG REQUIREMENTS:\n"
                "1. Generate 8-12 strategic hashtags that:\n"
                "   - Align with the marketing goals\n"
                "   - Follow campaign instructions\n"
                "   - Highlight product benefits from description\n"
                "   - Match campaign expectations\n"
                "   - Target the specific audience\n"
                "   - Include industry trends\n"
                "   - Drive engagement\n\n"
                "2. Include a mix of:\n"
                "   - Brand hashtags (product/company specific)\n"
                "   - Industry hashtags (sector/trend related)\n"
                "   - Audience hashtags (target demographic)\n"
                "   - Campaign hashtags (goal-oriented)\n"
                "   - Product feature hashtags\n"
                "   - Benefit-focused hashtags\n\n"
                "3. Best Practices:\n"
                "   - Keep hashtags concise and memorable\n"
                "   - Use relevant industry terminology\n"
                "   - Mix popular and niche hashtags\n"
                "   - Ensure hashtags are searchable\n"
                "   - Align with campaign goals\n"
                "   - Reflect product benefits\n\n"
                "4. Return ONLY a valid JSON array of strings\n"
                "5. Each hashtag must start with #\n"
                "6. NO markdown formatting or code blocks\n\n"
                "Example format:\n"
                "[\"#BrandName\", \"#IndustryTrend\", \"#TargetAudience\", \"#CampaignGoal\", \"#ProductFeature\"]\n\n"
                "YOUR RESPONSE:"
            )
        )
        hashtag_chain = LLMChain(llm=llm, prompt=hashtag_prompt)

        raw_output = await hashtag_chain.arun(
            keywords=", ".join(keywords),
            expectation=expectation,
            goals=", ".join(state["goals"]),
            instructions=state["instructions"],
            description=state["description"] or "Not provided"
        )
        logger.warning("Raw LLM output: %r", raw_output)
        
        try:
            expanded = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("LLM returned invalid JSON: %s", raw_output)
            raise e
        result = list(set(keywords + expanded))[:15]
        logger.info(result)
        return result
    except Exception as e:
        logger.warning("Hashtag expansion failed", error=str(e))
        return keywords

async def search_hashtags_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    
    try:
        hashtags = await generate_hashtags(state["company_keywords"], state["expectation"])
        state["hashtags"] = hashtags
        logger.info("Hashtags generated", agent_name=state["agent_name"], hashtags=hashtags)
    except Exception as e:
        state["error"] = f"Hashtag search failed: {str(e)}"
        logger.error("Hashtag search failed", agent_name=state["agent_name"], error=str(e))
    
    return state

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
async def fetch_tweets_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    start_time = datetime.now(timezone.utc)
    state["tweets"] = []

    try:
        logger.info("Starting tweet fetch process", 
                   agent_name=state["agent_name"],
                   hashtag_count=len(state["hashtags"]),
                   max_age_days=state["max_age_days"],
                   min_likes=state["min_likes"])

        for hashtag in state["hashtags"]:
            try:
                query_input = TweetQueryInput(
                    query=f"#{hashtag}",
                    minimum_tweets=10,
                    product="Top"
                )
                
                result = await fetch_tweets_by_query(query_input, state["db"])
                
                if result["status"] == "success":
                    state["tweets"].extend(result["tweets"])
                    logger.info(f"Processed hashtag {hashtag}", 
                              tweets_found=len(result["tweets"]),
                              total_tweets=len(result["tweets"]))
                
                await asyncio.sleep(random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME))
                
            except Exception as e:
                logger.error(f"Error processing hashtag {hashtag}", 
                           error=str(e),
                           stack_info=True)
                continue

        logger.info(
            "Tweets fetched successfully",
            agent_name=state["agent_name"],
            tweet_count=len(state["tweets"]),
            duration_sec=(datetime.now(timezone.utc) - start_time).total_seconds(),
            hashtags_processed=len(state["hashtags"])
        )

    except Exception as e:
        state["error"] = f"Tweet fetching failed: {str(e)}"
        logger.error(
            "Fetch failure",
            error=str(e),
            stack_info=True,
            agent_name=state.get("agent_name", "unknown"),
            duration_sec=(datetime.now(timezone.utc) - start_time).total_seconds()
        )

    return state

def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_input", validate_input_node)
    graph.add_node("search_hashtags", search_hashtags_node)
    graph.add_node("fetch_tweets", fetch_tweets_node)

    graph.set_entry_point("validate_input")
    
    graph.add_edge("validate_input", "search_hashtags")
    graph.add_edge("search_hashtags", "fetch_tweets")
    graph.add_edge("fetch_tweets", END)
    return graph.compile()

twitter_graph = create_graph()

@router.post("/twitter/twitter-agent", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def run_twitter_agent(input: TwitterAgentInput, db: AsyncSession = Depends(get_db)):
    try:
        initial_state = AgentState(
            agent_name=input.agent_name,
            goals=input.goals,
            instructions=input.instructions,
            description=input.description,
            expectation=input.expectation,
            target_audience=input.target_audience,
            company_keywords=input.company_keywords,
            min_likes=input.min_likes,
            max_age_days=input.max_age_days,
            hashtags=[],
            tweets=[],
            retries=0,
            error=None,
            db=db
        )
        
        result = await twitter_graph.ainvoke(initial_state)
        return result

    except Exception as e:
        logger.error("Twitter agent processing failed", agent_name=input.agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/twitter/fetch-tweets", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def fetch_tweets_by_query(input: TweetQueryInput, db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Starting tweet fetch process", 
                   query=input.query,
                   minimum_tweets=input.minimum_tweets,
                   product=input.product)

        twitter_client = await get_twitter_client()
        
        tweet_count = 0
        tweets = None
        results = []

        while tweet_count < input.minimum_tweets:
            try:
                if tweets is None:
                    logger.info(f"Getting initial tweets for query: {input.query}")
                    tweets = await twitter_client.search_tweet(input.query, product=input.product)
                else:
                    wait_time = random.randint(MIN_WAIT_TIME, MAX_WAIT_TIME)
                    logger.info(f"Getting next tweets after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    tweets = await tweets.next()

            except TooManyRequests as e:
                rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                wait_time = (rate_limit_reset - datetime.now(timezone.utc)).total_seconds()
                wait_time = max(wait_time + 60, 600)  # Minimum 10 minutes wait
                logger.warning(f"Rate limit reached. Waiting until {rate_limit_reset}", 
                             wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
                continue

            if not tweets:
                logger.info("No more tweets found")
                break

            for tweet in tweets:
                logger.info(tweet.id)
                tweet_count += 1
                hashtags = [word for word in tweet.text.split() if word.startswith('#')]
                tweet_data = {
                    "tweet_id": tweet.id,
                    "tweet_count": tweet_count,
                    "username": tweet.user.name,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "retweets": tweet.retweet_count,
                    "likes": tweet.favorite_count,
                    "hashtags": hashtags
                }
                results.append(tweet_data)

            logger.info(f"Got {tweet_count} tweets")

        logger.info("Tweet fetch completed", 
                   total_tweets=tweet_count,
                   query=input.query)

        return {
            "status": "success",
            "total_tweets": tweet_count,
            "tweets": results
        }

    except Exception as e:
        logger.error("Tweet fetch failed", 
                    error=str(e),
                    stack_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching tweets: {str(e)}"
        ) 