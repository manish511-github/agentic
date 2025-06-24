from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.ext.asyncio import AsyncSession
from .state import RedditAgentInput, AgentState
from .graph import reddit_graph
from app.database import get_db
import structlog

router = APIRouter()
logger = structlog.get_logger()

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
            keywords=[],
            posts=[],
            retries=0,
            error=None,
            db=db,
            llm=None
        )
        result = await reddit_graph.ainvoke(initial_state)
        return result
    except Exception as e:
        logger.error("Reddit agent processing failed", agent_name=input.agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 