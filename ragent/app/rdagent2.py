from __future__ import annotations

"""rdagent2.py - A self-contained asynchronous Reddit marketing agent.

The module exposes a single public coroutine `run_reddit_agent` which executes
these high-level steps:
1. Validate & normalise user input (`RedditAgentInput`)
2. Fetch recent Reddit posts that match supplied keywords (asyncpraw)
3. Ask Google Gemini to score relevance (0-100) and draft a reply for each post
4. Return structured output (`RedditAgentOutput`)

"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import asyncpraw  # Reddit API (async)
import google.genai as genai
import structlog
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv(
    "REDDIT_USER_AGENT", "agentic-rdagent2/0.1")


# DEBUG MODE toggles log-level and additional payload printing
# DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() in {
#     "true", "1", "yes"}
DEBUG_MODE = True

LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)
# Structlog configuration
# structlog.configure(
#     processors=[
#         structlog.processors.CallsiteParameterAdder(
#             parameters=[
#                 structlog.processors.CallsiteParameter.FILENAME,
#                 structlog.processors.CallsiteParameter.LINENO,
#             ]
#         ),
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.JSONRenderer()
#     ],
#     context_class=dict,
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
#     cache_logger_on_first_use=True,
# )

# Initialize logger
# logger = structlog.get_logger()
logger = logging.getLogger("rdagent2")


# ---------------------------------------------------------------------------
# Pydantic models for strong typing and validation
# ---------------------------------------------------------------------------

VALID_GOALS = {
    "increase brand awareness",
    "engage potential customers",
    "grow web traffic",
}


class RedditAgentInput(BaseModel):
    """Schema representing the user's request to the agent."""

    goals: List[str] = Field(..., description="Marketing goals to achieve.")
    instructions: str = Field(...,
                              description="How replies should be phrased.")
    expectation: str = Field(..., description="Desired marketing outcome.")
    keywords: List[str] = Field(...,
                                description="Target keywords to search for.")

    # Optional tuning knobs
    max_posts: int = Field(
        50, ge=1, le=200, description="Maximum posts to return.")
    min_upvotes: int = Field(
        0, ge=0, description="Ignore posts below this score.")
    max_age_days: int = Field(
        7, ge=1, le=365, description="Time-window for post age.")
    reply_temperature: float = Field(
        0.4, ge=0.0, le=1.0, description="Gemini temperature.")

    @field_validator("goals")
    @classmethod
    def _validate_goals(cls, v: List[str]):
        unknown = [g for g in v if g.lower() not in VALID_GOALS]
        if unknown:
            raise ValueError(
                f"Unsupported goal(s): {unknown}. Allowed: {sorted(VALID_GOALS)}")
        return [g.lower() for g in v]


class RedditPostResult(BaseModel):
    subreddit: str
    post_id: str
    post_title: str
    post_body: str
    post_url: str
    created_utc: datetime
    upvotes: int
    relevance: int = Field(ge=0, le=100)
    suggested_reply: str


class RedditAgentOutput(BaseModel):
    goals: List[str]
    posts: List[RedditPostResult]
    processed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Reddit helpers
# ---------------------------------------------------------------------------

def _get_reddit_client() -> asyncpraw.Reddit:
    """Instantiate and return an `asyncpraw.Reddit` client."""

    return asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _search_posts(
    reddit: asyncpraw.Reddit,
    keyword: str,
    *,
    max_results: int,
    min_upvotes: int,
    cutoff_ts: float,
) -> List[asyncpraw.models.Submission]:
    """Search Reddit for a keyword and return qualifying submissions."""

    logger.debug("Searching Reddit", extra={"keyword": keyword})
    submissions: List[asyncpraw.models.Submission] = []

    subreddit = await reddit.subreddit("all")
    try:
        search_iter = subreddit.search(
            keyword, limit=max_results, sort="relevance", syntax="lucene")
        async for submission in search_iter:
            if DEBUG_MODE:
                logger.debug(
                    f"Submission: {submission.id} {submission.title} {submission.score} {submission.created_utc} {submission.url}")
            if submission.score < min_upvotes:
                logger.debug(f"skipped submission: {submission.id}")
                continue
            if submission.created_utc < cutoff_ts:
                logger.debug(f"submission.created: {submission.created_utc}")
                logger.debug(
                    f"skipped submission due to time constraint {submission.id}")
                continue
            submissions.append(submission)
    except Exception as exc:
        logger.warning(f"Reddit search failed for '{keyword}': {exc}")
        raise  # triggers tenacity retry

    return submissions


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

class LLMResponseSchema(BaseModel):
    relevance: int
    reply: str


if not GOOGLE_API_KEY:
    logger.warning(
        "GOOGLE_API_KEY environment variable is not set - Gemini calls will fail.")

# Initialise Gemini once (thread-safe)
LLMClient = genai.Client(api_key=GOOGLE_API_KEY)


def _build_prompt(
    *,
    goals: List[str],
    expectation: str,
    instructions: str,
    post_title: str,
    post_body: str,
) -> str:
    """Compose a single-shot prompt asking Gemini for relevance & reply."""

    prompt = (
        "You are a seasoned marketing strategist. Evaluate the Reddit post below for its relevance to the given goals, "
        "then craft a concise helpful reply that adheres to the reply-style instructions.\n\n"
        f"Goals: {', '.join(goals)}\n"
        f"Expectation: {expectation}\n"
        f"Reply-Style Instructions: {instructions}\n\n"
        "Reddit Post:\n"
        f"Title: {post_title}\n"
        f"Body: {post_body[:2000]}\n\n"  # Truncate long posts
        "Return ONLY valid JSON exactly in this form (no markdown, no extra text):\n"
        "DO NOT include any backticks, markdown formatting, or extra explanation\n"
        "{\n  \"relevance\": <integer 0-100>,\n  \"reply\": <string>\n}"
    )
    return prompt


async def _gemini_call(prompt: str, *, temperature: float) -> Dict[str, Any]:
    """Call Gemini synchronously in a thread to avoid blocking the event loop."""

    def _invoke() -> Dict[str, Any]:
        response = LLMClient.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config={"temperature": temperature, "response_mime_type": "application/json",
                    "response_schema": LLMResponseSchema},
        )
        if DEBUG_MODE:
            logger.debug(f"Gemini raw response {response.text}", extra={
                         "response": response.text})
        try:
            data = json.loads(response.text.strip().replace(
                "```json", "").replace("```", ""))
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            if DEBUG_MODE:
                logger.debug(f"Gemini raw response {response.text}", extra={
                             "response": response.text})
            raise e
        if not isinstance(data, dict) or "relevance" not in data or "reply" not in data:
            raise ValueError("Gemini output missing required keys")
        data["relevance"] = max(0, min(100, int(round(data["relevance"]))))
        return data

    # Execute sync call in executor pool
    return await asyncio.to_thread(_invoke)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

async def run_reddit_agent(
    agent_input: RedditAgentInput,
    *,
    # Placeholder – real DB session to be wired later.
    db_session: Optional[Any] = None,
) -> RedditAgentOutput:
    """Execute the Reddit marketing agent end-to-end."""

    logger.info("Starting Reddit agent", extra={"goals": agent_input.goals})

    # ------------------------------------------------------------------
    # 1. Query Reddit for candidate posts
    # ------------------------------------------------------------------
    cutoff_ts = (datetime.now(tz=timezone.utc) -
                 timedelta(days=agent_input.max_age_days)).timestamp()
    logger.debug(f"cuttoff_ts: {cutoff_ts}")
    aggregate: Dict[str, asyncpraw.models.Submission] = {}

    async with _get_reddit_client() as reddit:
        search_tasks = [
            _search_posts(
                reddit,
                kw,
                max_results=agent_input.max_posts // max(
                    1, len(agent_input.keywords)) + 5,
                min_upvotes=agent_input.min_upvotes,
                cutoff_ts=cutoff_ts,
            )
            for kw in agent_input.keywords
        ]
        logger.info(f"Search tasks: {len(search_tasks)}")
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        logger.info(f"Search results: {len(search_results)}")

    for result in search_results:
        if isinstance(result, Exception):
            logger.error("Keyword search task failed", exc_info=result)
            continue
        for submission in result:
            aggregate[submission.id] = submission

    logger.info(f"Fetched candidate posts: {len(aggregate)}")

    if not aggregate:
        return RedditAgentOutput(goals=agent_input.goals, posts=[])

    # ------------------------------------------------------------------
    # 2. Evaluate relevance & craft replies via Gemini (concurrent, limited)
    # ------------------------------------------------------------------
    semaphore = asyncio.Semaphore(5)  # Concurrency limit for Gemini calls

    async def _process(sub: asyncpraw.models.Submission):
        async with semaphore:
            prompt = _build_prompt(
                goals=agent_input.goals,
                expectation=agent_input.expectation,
                instructions=agent_input.instructions,
                post_title=sub.title,
                post_body=sub.selftext,
            )

            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
            async def _call():
                return await _gemini_call(prompt, temperature=agent_input.reply_temperature)

            try:
                gemini_data = await _call()
            except Exception as exc:
                logger.error("Gemini evaluation failed", exc_info=exc)
                return None

            return RedditPostResult(
                subreddit=sub.subreddit.display_name,
                post_id=sub.id,
                post_title=sub.title,
                post_body=sub.selftext,
                post_url=f"https://www.reddit.com{sub.permalink}",
                created_utc=datetime.utcfromtimestamp(sub.created_utc),
                upvotes=sub.score,
                relevance=gemini_data["relevance"],
                suggested_reply=gemini_data["reply"],
            )

    tasks = [_process(s) for s in aggregate.values()]
    processed = await asyncio.gather(*tasks)

    results = [p for p in processed if p is not None]
    if DEBUG_MODE:
        logger.debug(f"Results: {results}")
    results.sort(key=lambda p: (p.relevance, p.upvotes), reverse=True)
    results = results[: agent_input.max_posts]

    logger.info("Gemini evaluation complete", extra={"returned": len(results)})

    # ------------------------------------------------------------------
    # 3. Persist results – future work
    # ------------------------------------------------------------------
    # TODO: When a DB layer becomes available, insert/update `results` here.
    # if db_session:
    #     await persist_to_db(db_session, results)

    return RedditAgentOutput(goals=agent_input.goals, posts=results)


# ---------------------------------------------------------------------------
# CLI sandbox (run `python -m ragent.app.rdagent2` for a quick demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _demo():
        sample_input = RedditAgentInput(
            goals=["increase brand awareness"],
            instructions="Be friendly, helpful and mention our SaaS CRM unobtrusively.",
            expectation="Locate discussions where small-business owners seek affordable CRM recommendations.",
            keywords=["affordable CRM", "small business CRM"],
            max_posts=100,
        )
        output = await run_reddit_agent(sample_input)
        print(json.dumps(output.model_dump(), indent=2, default=str))

    asyncio.run(_demo())
