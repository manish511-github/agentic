from ..state import AgentState
from ..reddit_client import get_reddit_client
from app.core.llm_client import get_llm, get_embedding_model
from app.core.vector_db import get_qdrant_client
import structlog
import json
from datetime import datetime
import asyncio
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from ...settings import settings
from ...utils.rate_limiter import reddit_limiter
from praw.models import SubredditHelper, Submission

logger = structlog.get_logger()


async def safe_gemini_call(llm, prompt):
    try:
        response = await llm.ainvoke(prompt)
        return response.content.lower()
    except Exception as e:
        logger.warning(f"Gemini call failed: {str(e)}")
        raise


async def delete_collection(collection_name):
    """
    Delete an existing collection in Qdrant
    """
    qdrant_client = get_qdrant_client()
    try:
        qdrant_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection {collection_name}")
    except Exception as e:
        if "not found" not in str(e).lower():
            logger.error(f"Error deleting collection: {str(e)}")
            raise


async def create_collection(collection_name):
    """
    Create a new collection in Qdrant
    """
    qdrant_client = get_qdrant_client()
    try:
        qdrant_client.create_collection(collection_name)
        logger.info(f"Created new collection {collection_name}")
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")


async def process_submission_batch(submissions: list[Submission], current_query: str, seen_post_ids: set[str], search_queries: list[str], state: AgentState):
    """
    Process a batch of submissions and return a list of posts
    """
    tasks: list[dict] = []
    for submission in submissions:
        if submission.id in seen_post_ids:
            continue
        seen_post_ids.add(submission.id)
        content = (submission.title + " " +
                   submission.selftext).lower()
        keyword_matches = sum(
            1 for kw in search_queries if kw.lower() in content)
        keyword_relevance = min(
            1.0, keyword_matches / max(1, len(state["company_keywords"])))
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
            "text": f"{submission.selftext}"
        }
        tasks.append(post_data)
    return tasks


async def process_query(query: str, all_subreddit: SubredditHelper):
    """
    Process a query and return a list of submissions and the query
    """
    submissions: list[Submission] = []
    try:
        # Honour global 100-req / 60-s limit
        async with reddit_limiter:
            async for submission in all_subreddit.search(
                query=query,
                sort="relevance",
                time_filter="month",
                limit=settings.posts_per_search
            ):
                submissions.append(submission)
                if len(submissions) >= settings.posts_per_search:
                    break
    except Exception as e:
        logger.warning(
            f"Search failed for query: {query}, error: {str(e)}",
            query=query,
            error=str(e))
    return submissions, query


async def search_posts_directly_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    try:
        reddit = await get_reddit_client()
        llm = get_llm()
        qdrant_client = get_qdrant_client()
        embedding_model = get_embedding_model()
        collection_name = "reddit_posts"

        # Delete and create collection
        try:
            await delete_collection(collection_name)
            await create_collection(collection_name)
        except Exception as e:
            logger.error(f"Error managing collection: {str(e)}")
            raise Exception(f"Error managing collection: {str(e)}")

        # Get company data
        company_data = {
            "agent_name": state.get("agent_name", ""),
            "goals": state.get("goals", []),
            "instructions": state.get("instructions", ""),
            "company_keywords": state.get("company_keywords", []),
            "description": state.get("description", ""),
            "target_audience": state.get("target_audience", ""),
            "expectation": state.get("expectation", ""),
            "keywords": state.get("keywords", "")
        }

        # Get search queries
        search_queries = company_data["keywords"]

        # Set up batching
        BATCH_SIZE = settings.qdrant_batch_size
        EMBEDDING_BATCH_SIZE = settings.embedding_batch_size
        current_batch = []
        posts = []
        seen_post_ids = set()

        async def generate_embeddings_batch(texts):
            try:
                embeddings = await embedding_model.aembed_documents(texts)
                return embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                return [None] * len(texts)

        # Process queries
        async with reddit as reddit_client:

            # Search the keywords generated by the LLM in all subreddits
            all_subreddit = await reddit_client.subreddit("all")
            query_tasks = [process_query(query, all_subreddit)
                           for query in search_queries]
            query_results = await asyncio.gather(*query_tasks)

            # Process the submissions
            for submissions, current_query in query_results:
                if not submissions:
                    continue
                # For all the submissions, count the number of keywords in the submission that are in the company keywords
                batch_posts = await process_submission_batch(submissions, current_query, seen_post_ids, search_queries, state)
                if not batch_posts:
                    continue

                # Embed the submissions in the Vector DB in batches
                for j in range(0, len(batch_posts), EMBEDDING_BATCH_SIZE):

                    # create a batch of posts to embed
                    embedding_batch = batch_posts[j:j + EMBEDDING_BATCH_SIZE]
                    texts = [post["text"] for post in embedding_batch]
                    embeddings = await generate_embeddings_batch(texts)

                    # embed the posts in the Vector DB
                    for post, embedding in zip(embedding_batch, embeddings):
                        if embedding is None:
                            continue
                        current_batch.append(PointStruct(
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
                        post.pop("text", None)
                        posts.append(post)
                    if len(current_batch) >= BATCH_SIZE:
                        try:
                            qdrant_client.upsert(
                                collection_name=collection_name,
                                points=current_batch
                            )
                            current_batch = []
                        except Exception as e:
                            logger.error(f"Error upserting batch: {str(e)}")
                            current_batch = []

            # Upsert the final batch
            if current_batch:
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=current_batch
                    )
                except Exception as e:
                    logger.error(f"Error upserting final batch: {str(e)}")

            # Semantic search the posts in the Vector DB
        try:
            query_text = f"""
            {company_data['description']}
            {', '.join(company_data['goals'])}
            {', '.join(search_queries)}
            {company_data['target_audience']}
            {company_data['expectation']}
            """
            query_embedding = await embedding_model.aembed_query(query_text)
            semantic_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=100,
                score_threshold=0.5
            )
            post_dict = {post["post_id"]: post for post in posts}
            semantic_posts = []
            for result in semantic_results:
                post_id = result.payload["submission_id"]
                if post_id in post_dict:
                    post = post_dict[post_id]
                    post["semantic_relevance"] = result.score
                    post["combined_relevance"] = (
                        0.8 * result.score +
                        0.2 * post["keyword_relevance"]
                    )
                    semantic_posts.append(post)
            semantic_posts.sort(
                key=lambda x: x["combined_relevance"], reverse=True)
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            posts.sort(key=lambda x: (
                x["keyword_relevance"], x["upvotes"]), reverse=True)
        state["posts"] = semantic_posts
        logger.info("Direct post search completed with semantic ranking", agent_name=state["agent_name"], posts_found=len(
            state["posts"]), unique_queries_used=search_queries)
    except Exception as e:
        state["error"] = f"Direct post search failed: {str(e)}"
        logger.error("Direct post search failed", error=str(e))
    return state
