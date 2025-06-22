from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.ext.asyncio import AsyncSession
from .models import TwitterAgentInput, TweetQueryInput, AgentState
from .graph_builder import twitter_graph
from .tweet_fetcher import fetch_tweets_by_query
from .x_agent_config import logger
from app.database import get_db

# Initialize FastAPI router
router = APIRouter()

@router.post("/twitter/twitter-agent", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def run_twitter_agent(input: TwitterAgentInput, db: AsyncSession = Depends(get_db)):
    """Run the Twitter agent workflow"""
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
async def fetch_tweets_endpoint(input: TweetQueryInput, db: AsyncSession = Depends(get_db)):
    """Fetch tweets by query"""
    try:
        logger.info("Starting tweet fetch process", 
                   query=input.query,
                   minimum_tweets=input.minimum_tweets,
                   product=input.product)

        result = await fetch_tweets_by_query(input, db)
        return result

    except Exception as e:
        logger.error("Tweet fetch failed", 
                    error=str(e),
                    stack_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching tweets: {str(e)}"
        ) 