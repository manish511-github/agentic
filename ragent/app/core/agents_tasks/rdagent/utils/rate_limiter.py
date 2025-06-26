from __future__ import annotations

from aiolimiter import AsyncLimiter

# Relative import to the settings singleton
from ..settings import settings

# One global limiter that replenishes 100 permits every 60 seconds (defaults)
# Values are pulled from environment or .env if present.
reddit_limiter: AsyncLimiter = AsyncLimiter(settings.reddit_rpm, 60)

__all__ = ["reddit_limiter"]
