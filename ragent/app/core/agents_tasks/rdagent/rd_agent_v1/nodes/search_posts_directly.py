from ..state import RedditAgentState
from ..reddit_client import get_reddit_client
from app.core.llm_client import get_llm, get_embedding_model
from app.core.vector_db import get_qdrant_client
from ...schemas import RedditPost
import structlog
import json
from datetime import datetime
import asyncio
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from ...settings import settings
from ...utils.rate_limiter import reddit_limiter
from asyncpraw.models import SubredditHelper, Submission
import time

logger = structlog.get_logger()


async def safe_gemini_call(llm, prompt):
    try:
        response = await llm.ainvoke(prompt)
        return response.content.lower()
    except Exception as e:
        logger.warning(f"Gemini call failed: {str(e)}")
        raise


async def delete_collection_with_retry(collection_name, max_retries=3):
    """
    Delete an existing collection in Qdrant with retry logic
    """
    for attempt in range(max_retries):
        try:
            qdrant_client = get_qdrant_client()
            qdrant_client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection {collection_name}")
            return
        except Exception as e:
            if "not found" in str(e).lower():
                logger.info(
                    f"Collection {collection_name} not found, skipping deletion")
                return

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Failed to delete collection (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"Error deleting collection after {max_retries} attempts: {str(e)}")
                raise


async def create_collection_with_retry(collection_name, max_retries=3):
    """
    Create a new collection in Qdrant with retry logic
    """
    for attempt in range(max_retries):
        try:
            qdrant_client = get_qdrant_client()
            # Specify the vector size according to your embedding model (default: 768)
            VECTOR_SIZE = 768
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            logger.info(f"Created new collection {collection_name}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Failed to create collection (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"Error creating collection after {max_retries} attempts: {str(e)}")
                raise


async def test_qdrant_connection(max_retries=3):
    """
    Test Qdrant connection with retry logic
    """
    for attempt in range(max_retries):
        try:
            qdrant_client = get_qdrant_client()
            qdrant_client.get_collections()
            logger.info("Qdrant connection test successful")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Qdrant connection test failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"Qdrant connection failed after {max_retries} attempts: {str(e)}")
                raise


async def process_submission_batch(submissions: list[Submission], current_query: str, seen_post_ids: set[str], search_queries: list[str], state: RedditAgentState):
    """
    Process a batch of submissions and return a list of posts
    """
    posts: list[RedditPost] = []
    for submission in submissions:
        if submission.id in seen_post_ids:
            continue
        seen_post_ids.add(submission.id)
        content = (submission.title + " " +
                   submission.selftext).lower()
        keyword_matches = sum(
            1 for kw in search_queries if kw.lower() in content)
        keyword_relevance = min(
            1.0, keyword_matches / max(1, len(state["keywords"])))

        post = RedditPost.from_reddit_submission(
            submission=submission,
            subreddit_name=submission.subreddit.display_name,
            keyword_relevance=keyword_relevance,
            matched_query=current_query
        )
        posts.append(post)
    return posts


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


async def search_posts_directly_node(state: RedditAgentState) -> RedditAgentState:
    if state.get("error"):
        return state

    try:
        # Test Qdrant connection first
        await test_qdrant_connection()

        reddit = await get_reddit_client()
        llm = get_llm()
        qdrant_client = get_qdrant_client()
        embedding_model = get_embedding_model()
        collection_name = "reddit_posts"

        # Delete and create collection with retry logic
        try:
            await delete_collection_with_retry(collection_name)
            await create_collection_with_retry(collection_name)
        except Exception as e:
            logger.error(f"Error managing collection: {str(e)}")
            raise Exception(f"Error managing collection: {str(e)}")

        # Get company data
        company_data = {
            "agent_name": state.get("agent_name", ""),
            "goals": state.get("goals", []),
            "instructions": state.get("instructions", ""),
            "description": state.get("description", ""),
            "target_audience": state.get("target_audience", ""),
            "expectation": state.get("expectation", ""),
            "keywords": state.get("keywords", "")
        }

        # Get search queries
        search_queries = state.get("generated_queries", [])

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
                semaphore = asyncio.Semaphore(EMBEDDING_BATCH_SIZE)

                async def embed_and_upsert(embedding_batch):
                    texts = [post.text for post in embedding_batch]
                    async with semaphore:
                        embeddings = await generate_embeddings_batch(texts)
                        batch_points = []
                        batch_posts = []
                        for post, embedding in zip(embedding_batch, embeddings):
                            if embedding is None:
                                continue
                            batch_points.append(PointStruct(
                                id=abs(hash(post.post_id)) % (2**63),
                                vector=embedding,
                                payload=post.to_vector_payload()
                            ))
                            batch_posts.append(post)
                        if batch_points:
                            try:
                                qdrant_client.upsert(
                                    collection_name=collection_name,
                                    points=batch_points
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error upserting batch: {str(e)}")
                        return batch_posts

                embed_tasks = []
                for j in range(0, len(batch_posts), EMBEDDING_BATCH_SIZE):
                    embedding_batch = batch_posts[j:j + EMBEDDING_BATCH_SIZE]
                    embed_tasks.append(embed_and_upsert(embedding_batch))
                embed_results = await asyncio.gather(*embed_tasks)

                for batch_posts_result in embed_results:
                    posts.extend(batch_posts_result)
                current_batch = []  # Clear current_batch since upserts are handled in tasks

            # Semantic search the posts in the Vector DB
        try:
            query_text = f"""
            {company_data['description']}
            {', '.join(company_data['goals'])}
            {', '.join(search_queries)}
            {company_data['target_audience']}
            {company_data['expectation']}
            {', '.join(company_data['keywords'])}
            """
            query_embedding = await embedding_model.aembed_query(query_text)
            semantic_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=100,
                score_threshold=0.5
            )
            post_dict = {post.post_id: post for post in posts}
            semantic_posts = []
            for result in semantic_results:
                post_id = result.payload["submission_id"]
                if post_id in post_dict:
                    post = post_dict[post_id]
                    # Update the post object with semantic relevance scores
                    post.semantic_relevance = result.score
                    post.combined_relevance = (
                        0.8 * result.score +
                        0.2 * post.keyword_relevance
                    )
                    semantic_posts.append(post)
            semantic_posts.sort(
                key=lambda x: x.combined_relevance, reverse=True)
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            posts.sort(key=lambda x: (
                x.keyword_relevance, x.upvotes), reverse=True)
        state["direct_posts"] = semantic_posts
        logger.info("Direct post search completed with semantic ranking", agent_name=state["agent_name"], posts_found=len(
            state["posts"]), unique_queries_used=search_queries)
    except Exception as e:
        state["error"] = f"Direct post search failed: {str(e)}"
        logger.error("Direct post search failed", error=str(e))
    return state
