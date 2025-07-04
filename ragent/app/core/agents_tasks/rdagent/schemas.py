from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class RedditPost(BaseModel):
    """Comprehensive Reddit post schema used across the Reddit agent."""
    
    # Core Reddit API fields
    subreddit: str = Field(..., description="Subreddit name without the r/ prefix")
    post_id: str = Field(..., description="Reddit submission id")
    post_title: str = Field(..., description="Submission title")
    post_body: str = Field(..., description="Submission self-text (may be empty)")
    post_url: str = Field(..., description="Canonical Reddit URL")
    upvotes: int = Field(..., description="Number of upvotes (submission score)")
    comment_count: int = Field(..., description="Number of comments")
    created: str = Field(..., description="Submission creation date in ISO format")
    
    # Processing and analysis fields
    text: str = Field(..., description="Combined text (title + body) for embeddings")
    keyword_relevance: float = Field(default=0.0, ge=0, le=1, description="Keyword relevance score")
    semantic_relevance: Optional[float] = Field(default=None, ge=0, le=1, description="Semantic relevance score from vector search")
    combined_relevance: Optional[float] = Field(default=None, ge=0, le=1, description="Combined relevance score")
    llm_relevance: Optional[float] = Field(default=None, ge=0, le=1, description="LLM relevance score")
    
    # Query and processing tracking
    matched_query: Optional[str] = Field(default=None, description="Query that matched this submission")
    sort_method: Optional[str] = Field(default=None, description="Sort method used to find this post (hot, new, top, etc.)")
    
    # Legacy fields for backward compatibility
    relevance_score: Optional[float] = Field(default=None, ge=0, le=1, description="Legacy relevance score")
    sentiment_score: Optional[float] = Field(default=None, ge=-1, le=1, description="Legacy sentiment score")

    class Config:
        # Allow extra fields for flexibility during transition
        extra = "forbid"
        
    def to_vector_payload(self) -> dict:
        """Convert to format suitable for vector database storage."""
        return {
            "submission_id": self.post_id,
            "title": self.post_title,
            "content": self.post_body,
            "url": self.post_url,
            "created_utc": datetime.fromisoformat(self.created).timestamp(),
            "subreddit": self.subreddit,
            "score": self.upvotes,
            "num_comments": self.comment_count,
            "keyword_relevance": self.keyword_relevance,
        }
    
    @classmethod
    def from_reddit_submission(cls, submission, subreddit_name: str, keyword_relevance: float = 0.0, 
                             matched_query: Optional[str] = None, sort_method: Optional[str] = None):
        """Create RedditPost from asyncpraw submission object."""
        created_iso = datetime.utcfromtimestamp(submission.created_utc).isoformat()
        text_content = f"{submission.title} {submission.selftext}"
        
        return cls(
            subreddit=subreddit_name,
            post_id=submission.id,
            post_title=submission.title,
            post_body=submission.selftext,
            post_url=f"https://www.reddit.com{submission.permalink}",
            upvotes=submission.score,
            comment_count=submission.num_comments,
            created=created_iso,
            text=text_content,
            keyword_relevance=keyword_relevance,
            matched_query=matched_query,
            sort_method=sort_method
        )


class RedditAgentOutput(BaseModel):
    """Final structured response returned by the Reddit agent."""

    agent_name: str
    goals: List[str]
    instructions: str
    posts: List[RedditPost]
    direct_posts: Optional[List[RedditPost]] = Field(default=None, description="Posts from direct search")
    subreddit_posts: Optional[List[RedditPost]] = Field(default=None, description="Posts from subreddit search")
