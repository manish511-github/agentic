from ..state import RedditAgentState
from ..reddit_client import get_reddit_client
from app.core.llm_client import get_embedding_model, get_llm
from app.core.vector_db import get_collection_name, get_qdrant_client
from ...schemas import RedditPost
import structlog
from datetime import datetime
import asyncio
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from ...settings import settings
from ...utils.rate_limiter import reddit_limiter
from ...utils.querymaker import create_OR_query_in_batch
from asyncpraw import Reddit
from typing import List, Dict

logger = structlog.get_logger()

# Concurrency guard so we don't spin up an unreasonable number of coroutines.
semaphore = asyncio.Semaphore(settings.max_concurrency)


async def fetch_basic_post_nodes(state: RedditAgentState) -> RedditAgentState:
    if state.get("error"):
        return state
    state["posts"] = []
    try:
        llm = get_llm()
        async with await get_reddit_client() as reddit:
            posts = []
            seen_post_ids = set()
            semaphore = asyncio.Semaphore(settings.max_concurrency)
            async def process_subreddit(subreddit_name: str):
                async with semaphore:
                    subreddit_posts = []
                    try:
                        subreddit = await reddit.subreddit(subreddit_name)
                        await subreddit.load()
                        logger.info(
                            f"Successfully loaded subreddit {subreddit_name}")
                    except Exception as e:
                        logger.error(
                            f"Failed to load subreddit {subreddit_name}: {str(e)}")
                        return subreddit_posts

                async def fetch_listing(listing, sort_method):
                    sub_posts = []
                    try:
                        async with reddit_limiter:
                            async for submission in listing:
                                if submission.id in seen_post_ids:
                                    continue
                                seen_post_ids.add(submission.id)
                                post = RedditPost.from_reddit_submission(
                                    submission=submission,
                                    subreddit_name=subreddit_name,
                                    sort_method=sort_method
                                )
                                sub_posts.append(post)
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch posts from {sort_method} in {subreddit_name}: {str(e)}")
                    return sub_posts
                # Build listing tasks dynamically based on configured categories.
                tasks = []
                for cat in settings.categories:
                    try:
                        if cat == "hot":
                            listing = subreddit.hot(
                                limit=settings.posts_per_search)
                        elif cat == "new":
                            listing = subreddit.new(
                                limit=settings.posts_per_search)
                        elif cat == "top":
                            # Default to the past week when using "top" to keep results fresh
                            listing = subreddit.top(
                                "week", limit=settings.posts_per_search)
                        elif cat == "rising":
                            listing = subreddit.rising(
                                limit=settings.posts_per_search)
                        else:
                            logger.warning(
                                f"Unsupported listing category '{cat}' – skipping")
                            continue
                        tasks.append(fetch_listing(listing, cat))
                    except Exception as e:
                        logger.warning(
                            f"Failed to build listing for category '{cat}': {str(e)}")
                        continue
                listing_results = await asyncio.gather(*tasks)
                for sublist in listing_results:
                    subreddit_posts.extend(sublist)
                logger.info(
                    f"Processed subreddit {subreddit_name} with {len(subreddit_posts)} posts")
                return subreddit_posts
            subreddit_names = state.get("subreddits", [])
            posts = []
            subreddit_names = subreddit_names[:30]  # Keep only the first 10 subreddit names
            for name in subreddit_names:
                try:
                    res = await process_subreddit(name)
                    posts.extend(res)
                except Exception as e:
                    logger.error(
                        f"Error processing subreddit {name}", error=str(e))
                    continue
            # Batchify utility

            def batchify(lst, batch_size):
                for i in range(0, len(lst), batch_size):
                    yield lst[i:i + batch_size]
            batch_size = settings.llm_batch_size
            all_scores = []
            import json
            semaphore = asyncio.Semaphore(10)

            async def score_batch(batch):
                prompt = (
                    f"Score the relevance (0-1, float) of each post for a product with the following details:\n"
                    f"Product Description: {state['description']}\n"
                    f"Goals: {', '.join(state['goals'])} (ensure posts support these marketing objectives)\n"
                    f"Expectation: {state['expectation']} (posts should match this content focus)\n"
                    f"Target Audience: {state['target_audience']}\n"
                    f"Keywords: {', '.join(state['keywords'])}\n\n"
                    "Posts:\n" +
                    "\n".join([
                        f"Post {i}: {c.post_title}\n{c.post_body[:1000]}"
                        for i, c in enumerate(batch, 1)
                    ]) +
                    f"\n\nOUTPUT REQUIREMENTS:\n"
                    f"- Return ONLY a valid JSON array of floats (0-1), one for each post above.\n"
                    f"- The array must have exactly {len(batch)} numbers, in the same order as the posts.\n"
                    f"- DO NOT include any markdown, code block, backticks, or extra text.\n"
                    f"- DO NOT include any explanation or formatting.\n"
                    f"- Your response MUST be a plain JSON array, e.g. [0.7, 0.3, 0.9, 0.5] and nothing else.\n"
                )
                async with semaphore:
                    try:
                        response = await llm.ainvoke(prompt)
                        scores_raw = response.content if hasattr(response, 'content') else response
                        logger.info(f"Raw LLM response: {scores_raw}")
                        cleaned = scores_raw.strip()
                        if cleaned.startswith('```json'):
                            cleaned = cleaned[7:]
                        if cleaned.startswith('```'):
                            cleaned = cleaned[3:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3]
                        cleaned = cleaned.strip()
                        if not cleaned:
                            raise ValueError("LLM returned empty response")
                        try:
                            scores = json.loads(cleaned)
                        except Exception as parse_err:
                            logger.error("Failed to parse LLM response as JSON", error=str(parse_err), response=scores_raw)
                            scores = [0.0] * len(batch)
                        if not isinstance(scores, list) or len(scores) != len(batch):
                            logger.error("LLM response is not a list of correct length", response=scores_raw)
                            scores = [0.0] * len(batch)
                        logger.info(f"Processed batch of {len(batch)} posts with {len(scores)} scores returned")
                        return scores
                    except Exception as e:
                        logger.error("LLM batch scoring failed", error=str(e))
                        return [0.0] * len(batch)

            batch_tasks = [score_batch(batch) for batch in batchify(posts, batch_size)]
            batch_results = await asyncio.gather(*batch_tasks)
            for scores in batch_results:
                all_scores.extend(scores)
            for post, score in zip(posts, all_scores):
                post.llm_relevance = score
            logger.info("Fetch posts completed", agent_name=state["agent_name"], posts_count=(posts))
            state["subreddit_posts"] = posts
    except Exception as e:
        state["error"] = f"Post fetching failed: {str(e)}"
        logger.error(
            "Fetch failure",
            error=str(e),
            stack_info=True,
            agent_name=state.get("agent_name", "unknown")
        )
    return state



async def process_subreddit(subreddit_name: str, state: RedditAgentState):
    async with await get_reddit_client() as reddit:
        async with semaphore:
            subreddit_posts = []
            try:
                logger.info(f"Processing subreddit: {subreddit_name}")
                try:
                    subreddit = await reddit.subreddit(subreddit_name)
                    await subreddit.load()
                    logger.info(
                        f"Successfully loaded subreddit: {subreddit_name}")
                except Exception as e:
                    logger.error(
                        f"Failed to load subreddit {subreddit_name}: {str(e)}")
                    return subreddit_posts
                # Build batched OR queries from company keywords
                keyword_batch_size = settings.keyword_batch_size
                search_queries = list(
                    create_OR_query_in_batch(
                        state["keywords"],
                        keyword_batch_size,
                        quote=True,
                    )
                )
                logger.info(
                    f"Generated search queries for {subreddit_name}: {search_queries}")
                for search_query in search_queries:
                    try:
                        logger.info(
                            f"Searching {subreddit_name} with query: {search_query}")
                        async with reddit_limiter:
                            async for submission in subreddit.search(
                                    search_query,
                                    sort="relevance",
                                    time_filter="month",
                                    limit=settings.posts_per_search):
                                if submission.id in state.get("seen_post_ids"):
                                    continue
                                state["seen_post_ids"].add(submission.id)
                                content = (submission.title + " " +
                                            submission.selftext).lower()
                                keyword_matches = sum(
                                    1 for kw in state["keywords"] if kw.lower() in content)
                                keyword_relevance = min(
                                    1.0, keyword_matches / max(1, len(state["keywords"])))
                                post = RedditPost.from_reddit_submission(
                                    submission=submission,
                                    subreddit_name=subreddit_name,
                                    keyword_relevance=keyword_relevance
                                )
                                subreddit_posts.append(post)
                                logger.info(
                                    f"Found post in {subreddit_name}: {submission.title}")
                                # Emit a terse per-post log line
                                logger.info(
                                    "embedded_post",
                                    post_id=post.post_id,
                                    subreddit=post.subreddit,
                                )
                    except Exception as e:
                        logger.warning(
                            f"Search failed for query '{search_query}' in {subreddit_name}: {str(e)}")
                        continue
                logger.info(
                    f"Completed processing {subreddit_name} (embedding & persistence)",
                    posts_found=len(subreddit_posts),
                )

                # ---------------------------------------------
                #  Embedding & persistence happen here so that
                #  IO bound work (Reddit fetch) can overlap with
                #  CPU / network bound work (embedding + Qdrant).
                # ---------------------------------------------
                await process_subreddit_posts(subreddit_posts, state)

                logger.info(
                    f"Completed processing {subreddit_name}",
                    posts_found=len(subreddit_posts),
                    keyword_matches=sum(
                        1 for p in subreddit_posts if p["keyword_relevance"] > 0),
                    # Duration omitted for brevity
                )
            except Exception as e:
                logger.error(f"Error processing {subreddit_name}: {str(e)}",
                                subreddit=subreddit_name,
                                error=str(e))
            return subreddit_posts
        
async def process_subreddit_posts(subreddit_posts: List[RedditPost], state: RedditAgentState):
    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()
    collection_name = get_collection_name(state["agent_name"],"rdagent_v1")
    for idx in range(0, len(subreddit_posts), settings.embedding_batch_size):
        batch = subreddit_posts[idx: idx +
                                settings.embedding_batch_size]
        batch_texts = [p.text for p in batch]
        try:
            embeddings = await embedding_model.aembed_documents(batch_texts)
            points = []
            for post, emb in zip(batch, embeddings):
                points.append(
                    PointStruct(
                        id=abs(hash(post.post_id)) % (
                            2**63),
                        vector=emb,
                        payload=post.to_vector_payload(),
                    )
                )
            # Persist the vectors
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )
        except Exception as err:
            logger.error(
                "embedding_or_upsert_failed",
                error=str(err),
            )

async def fetch_posts_node(state: RedditAgentState) -> RedditAgentState:
    if state.get("error"):
        logger.warning("Skipping fetch_posts_node due to existing error",
                       agent_name=state.get("agent_name", "unknown"),
                       error=state.get("error"))
        return state

    start_time = datetime.utcnow()
    state["posts"] = []

    try:
        valid_goals = ["increase brand awareness",
                       "engage potential customers", "grow web traffic"]
        if not all(g in valid_goals for g in state["goals"]):
            state["error"] = f"Invalid goals: {[g for g in state['goals'] if g not in valid_goals]}"
            logger.error("Invalid goals",
                         agent_name=state["agent_name"],
                         invalid_goals=[g for g in state["goals"]
                                        if g not in valid_goals],
                         valid_goals=valid_goals)
            return state

        logger.info("Starting fetch_posts_node",
                    agent_name=state["agent_name"],
                    goals=state["goals"],
                    keywords=state["keywords"],
                    subreddits=state["subreddits"])
        
        # Initialize Qdrant client and embedding model
        try:
            qdrant_client = get_qdrant_client()
            embedding_model = get_embedding_model()
            logger.info("Initialized Qdrant client and embedding model")

            collection_name = get_collection_name(state["agent_name"],"rdagent_v1")
            # Get Reddit client without context manager to avoid premature session closure
            reddit = await get_reddit_client()
        except Exception as e:
            logger.error(f"Error initializing Qdrant client and embedding model: {str(e)}")
            raise

        # Create collection if it doesn't exist
        try:
            try:
                qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection {collection_name}")
            except Exception as e:
                if "not found" not in str(e).lower():
                    logger.error(f"Error deleting collection: {str(e)}")
                    raise
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created new collection {collection_name}")
        except Exception as e:
            logger.error(f"Error managing collection: {str(e)}")
            raise

        try:
            BATCH_SIZE = settings.max_subreddits_per_batch
            posts = []
            total_posts_processed = 0
            total_keyword_matches = 0

            # ---------------- Batch over subreddits ----------------
            for i in range(0, len(state["subreddits"]), BATCH_SIZE):
                batch = state["subreddits"][i:i + BATCH_SIZE]
                batch_start_time = datetime.utcnow()
                logger.info(
                    f"Processing batch {i//BATCH_SIZE + 1}",
                    batch_size=len(batch),
                    subreddits=batch,
                )
                try:
                    results = await asyncio.gather(
                        *(process_subreddit(name, state) for name in batch),
                        return_exceptions=True,
                    )
                    for result in results:
                        if isinstance(result, list):
                            posts.extend(result)
                            total_posts_processed += len(result)
                        elif isinstance(result, Exception):
                            logger.error("batch_processing_error",
                                         error=str(result))
                    logger.info(
                        f"Completed batch {i//BATCH_SIZE + 1}",
                        duration_sec=(datetime.utcnow() -
                                      batch_start_time).total_seconds(),
                        posts_in_batch=len(posts) - total_posts_processed,
                    )
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue
            try:
                logger.info("Starting semantic search")
                query_text = f"""
                {state['description']}
                {', '.join(state['goals'])}
                {', '.join(state['keywords'])}
                {state['target_audience']}
                {state['expectation']}
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
                        post.semantic_relevance = result.score
                        post.combined_relevance = (
                            0.8 * result.score +
                            0.2 * post.keyword_relevance
                        )
                        semantic_posts.append(post)
                semantic_posts.sort(
                    key=lambda x: x.combined_relevance, reverse=True)
                if state.get("posts"):
                    existing_post_ids = {post.post_id
                                         for post in state["posts"]}
                    new_posts = [
                        post for post in semantic_posts if post.post_id not in existing_post_ids]
                    state["posts"].extend(new_posts)
                else:
                    state["posts"] = semantic_posts
                logger.info("Semantic search completed",
                            total_posts=len(semantic_posts),
                            avg_semantic_score=sum(p.semantic_relevance for p in semantic_posts) / len(semantic_posts) if semantic_posts else 0)
            except Exception as e:
                logger.error(f"Semantic search failed: {str(e)}",
                             error=str(e))
                state["posts"] = sorted(
                    posts,
                    key=lambda x: (x.keyword_relevance, x.upvotes),
                    reverse=True
                )
                logger.info("Falling back to keyword-based sorting",
                            total_posts=len(state["posts"]))
            logger.info(
                "Posts fetched and analyzed successfully",
                agent_name=state["agent_name"],
                post_count=len(state["posts"]),
                total_posts_processed=total_posts_processed,
                total_keyword_matches=total_keyword_matches,
                duration_sec=(datetime.utcnow() -
                              start_time).total_seconds()
            )
        finally:
            # Ensure Reddit client is properly closed
            await reddit.close()
    except Exception as e:
        state["error"] = f"Post fetching failed: {str(e)}"
        logger.error(
            "Fetch failure",
            error=str(e),
            stack_info=True,
            agent_name=state.get("agent_name", "unknown")
        )
    return state
