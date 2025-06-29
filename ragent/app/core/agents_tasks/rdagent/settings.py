from pydantic_settings import BaseSettings


class RedditAgentSettings(BaseSettings):
    """Runtime-configurable knobs for the Reddit agent.

    All values can be overridden via environment variables or a .env file so
    that the behaviour of the agent (throughput, batch sizes, etc.) can be
    tuned without touching code.
    """

    # --- Reddit specific -------------------------------------------------
    # Maximum number of Reddit HTTP requests permitted per minute (free tier)
    reddit_rpm: int = 100
    # The maximum number of subreddits to process in a single batch
    # This is used when we are fetching posts from subreddits
    max_subreddits_per_batch: int = 10
    # The maximum number of concurrent producer tasks (subreddit / query fetch)
    max_concurrency: int = 20

    # How many posts to request for each search/listing call
    posts_per_search: int = 20

    # The post listing categories (order matters) to scan for each subreddit.
    # Valid values: "hot", "new", "top", "rising", etc.
    # Provide a comma-separated list via the RDAGENT_CATEGORIES env variable to override.
    categories: list[str] = ["hot", "new", "top"]

    # --- Embedding / Vector DB ------------------------------------------
    embedding_batch_size: int = 64     # documents per embedding request
    qdrant_batch_size: int = 200       # vectors per upsert

    # --- LLM invocation --------------------------------------------------
    llm_batch_size: int = 20           # posts per relevance-scoring prompt
    llm_concurrency: int = 5           # simultaneous LLM calls

    # --- Misc ------------------------------------------------------------
    max_total_posts: int = 1000        # safety valve to avoid runaway runs

    class Config:
        env_prefix = "RDAGENT_"        # e.g. RDAGENT_REDDIT_RPM=60
        env_file = ".env"


# Instantiate a singleton that code can import directly
settings = RedditAgentSettings()
