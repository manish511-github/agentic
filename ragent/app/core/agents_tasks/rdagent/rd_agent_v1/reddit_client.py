from asyncpraw import Reddit
import os

_reddit_client = None

async def get_reddit_client() -> Reddit:
    global _reddit_client
    if _reddit_client is None:
        _reddit_client = Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )
    return _reddit_client


