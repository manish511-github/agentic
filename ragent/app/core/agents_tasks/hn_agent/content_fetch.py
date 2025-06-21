import datetime
from http.client import HTTPException
import aiohttp
import structlog
from typing import Dict, Any, List, Optional
from qdrant_client.http import models
from .models import AgentState, HNStory
from app.core.vector_db import get_qdrant_client, get_collection_name, ensure_collection_exists
from app.core.llm_client import get_embedding_model
import asyncio

logger = structlog.get_logger()

# We can also use  Semaphore for limiting concurrent request
# semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

# async def safe_fetch(story_id):
#     async with semaphore:
#         return await fetch_hn_story(story_id)

async def fetch_hn_story(item_id: int) -> Optional[Dict]:
    """
    Fetch a single Hacker News item (story or comment) by ID from the Firebase API.
    This is used for fetching initial story details and top-level comment IDs.
    """
    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status() # Raise an exception for HTTP errors
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch HN item {item_id} from Firebase: {e}")
            return None



async def fetch_story_ids(feed_type="topstories", limit=100):
    """
    Fetch a list of story IDs from a given HN feed type (e.g., "topstories").
    Uses Firebase API.
    """
    url = f"https://hacker-news.firebaseio.com/v0/{feed_type}.json"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                ids = await resp.json()
                return ids[:limit]
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch story IDs from {feed_type} feed: {e}")
            return []


async def algolia_search(query: str, min_score: int, max_age_days: int, item_type: str, limit: int = 100):
    """
    Search Hacker News stories or comments using the Algolia API for full-text search.
    Filters by minimum score and maximum age.
    :param query: The search query string.
    :param min_score: Minimum points (score) for a story/comment.
    :param max_age_days: Maximum age of the item in days.
    :param item_type: 'story' or 'comment' to specify what to search for.
    :param limit: Number of hits to return.
    """
    # Algolia uses 'points' for score and 'created_at_i' for Unix timestamp
    now = datetime.datetime.utcnow().timestamp()
    min_time = now - max_age_days * 86400 # Convert days to seconds

    # Construct Algolia API URL for search
    url = (
        f"https://hn.algolia.com/api/v1/search?"
        f"query={query}&tags={item_type}&hitsPerPage={limit}"
        # f"&numericFilters=points>={min_score},created_at_i>={int(min_time)}"
    )

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("hits", [])
        except aiohttp.ClientError as e:
            logger.error(f"Algolia search failed for query '{query}' (type: {item_type}): {e}")
            return []
        

async def fetch_and_store_content_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to fetch stories from multiple sources and store in vector database.
    """
    agent_input = state["agent_input"]
    agent_name = agent_input.agent_name
    queries = state["expanded_queries"]
    
    # Get collection name for this agent
    collection_name = get_collection_name(agent_name, "hackernews")
    
    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()
    
    try:
        try:
            qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted existing Qdrant collection: {collection_name}")
        except Exception as e:
            logger.info(f"Qdrant collection '{collection_name}' did not exist or could not be deleted: {e}")
        qdrant_client.create_collection(collection_name=collection_name,vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))        
        logger.info(f"Created new Qdrant collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create/configure Qdrant collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set up Qdrant collection: {str(e)}")
    
    all_items_to_store = {}
    all_story_ids = []
    for feed in ["topstories", "newstories", "beststories", "showstories", "askstories"]:
        ids = await fetch_story_ids(feed, limit=60)
        all_story_ids.extend(ids)
    all_story_ids = list(set(all_story_ids))

    # Fetch from simple HN firebase Api

    BATCH_SIZE = 50
    firebase_stories = []
    for i in range(0, len(all_story_ids), BATCH_SIZE):
        batch_ids = all_story_ids[i:i+BATCH_SIZE]
        logger.info(f"Fetching stories for batch {i//BATCH_SIZE + 1} of {len(all_story_ids)//BATCH_SIZE + 1}")
        fetch_firebase_tasks = [fetch_hn_story(story_id) for story_id in batch_ids]
        try:
            batch_results = await asyncio.gather(*fetch_firebase_tasks, return_exceptions=True)
            logger.info(f"Fetched batch {i//BATCH_SIZE + 1} of {len(all_story_ids)//BATCH_SIZE + 1} stories")
            firebase_stories.extend(batch_results)
        except Exception as e:
            logger.error(f"Failed to fetch batch {i//BATCH_SIZE + 1} of {len(all_story_ids)//BATCH_SIZE + 1} stories: {e}")

    for story in firebase_stories:
        if not story:
            continue
        normalized_story = {
            "id": int(story["id"]),
            "type": "story",
            "title": story.get("title"),
            "text": story.get("text"),
            "url": story.get("url"),
            "score": story.get("score", 0),
            "time": story.get("time", 0),
            "relevance": 0.0
        }
        all_items_to_store[str(story["id"])] = normalized_story
    logger.info(f"Fetched {len(firebase_stories)} stories from Firebase and added to processing queue.")
    
    # Fetch from  HN Algolia Api

    min_score = agent_input.min_score if hasattr(agent_input, 'min_score') else 0
    max_age_days = agent_input.max_age_days if hasattr(agent_input, 'max_age_days') else 30
    algolia_hits = []
    ALGOLIA_BATCH_SIZE = 5
    # Prepare all tasks
    algolia_hits = []
    for i in range(0, len(queries), ALGOLIA_BATCH_SIZE):
        batch_queries = queries[i:i+ALGOLIA_BATCH_SIZE]

        logger.info("batch_queries" + str(batch_queries))
        logger.info(f"Fetching Algolia batch {i//ALGOLIA_BATCH_SIZE + 1} of {len(queries)//ALGOLIA_BATCH_SIZE + 1}")
        # For each query in the batch, run both story and comment searches in parallel
        batch_tasks = [
            asyncio.gather(
                algolia_search(q, min_score, max_age_days, "story", limit=100),
                algolia_search(q, 0, max_age_days, "comment", limit=100)
            )
            for q in batch_queries
        ]
        # Gather results for the batch (each result is a tuple: (stories, comments))
        batch_results = await asyncio.gather(*batch_tasks)
        # Flatten and extend
        for stories, comments in batch_results:
            for story in stories:
                story["type"] = "story"
                algolia_hits.append(story)
            for comment in comments:
                comment["type"] = "comment"
                algolia_hits.append(comment)
        logger.info(f"Fetched Algolia batch {i//ALGOLIA_BATCH_SIZE + 1} of {len(queries)//ALGOLIA_BATCH_SIZE + 1}")

    # Yes! we got data from algolia sucessfully -> lets get it into structure

    for hit in algolia_hits:
        item_id = str(hit.get("objectID") or hit.get("id"))
        if not item_id:
            continue

        if item_id not in all_items_to_store:
            item_type = hit.get("type", "story")
        
            payload = {
                "id": int(item_id),
                "type": item_type,
                "title": hit.get("title") if item_type == "story" else None,
                "story_text": hit.get("story_text") if item_type == "story" else None,
                "comment_text": hit.get("comment_text") if item_type == "comment" else None,
                "url": hit.get("story_url"),
                "score": hit.get("points", 0) if hit.get("points") is not None else hit.get("score", 0),
                "time": hit.get("created_at_i", 0) if hit.get("created_at_i") is not None else hit.get("time", 0),
                "story_id": int(hit.get("story_id")) if hit.get("story_id") else (int(item_id) if item_type == "story" else None),
                "parent_story_id": int(hit.get("story_id")) if item_type == "comment" and hit.get("story_id") else None,
                "children": hit.get("children", []),
                "relevance": 0.0
            }
            # if item_type == "story" and payload["score"] >= agent_input.min_score:
            if item_type == "story":
                all_items_to_store[item_id] = payload
            elif item_type == "comment":
                all_items_to_store[item_id] = payload
    logger.info(f"Fetched {len(algolia_hits)} hits from Algolia and added to processing queue. Total unique items: {len(all_items_to_store)}")

    # Create Text embedding in batches and store in Quadrant 
    BATCH_SIZE = 500
    CONCURRENCY_LIMIT = 5  # Tune this based on your resources
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    items_to_process = list(all_items_to_store.values())

    async def process_batch(batch, embedding_model, qdrant_client, collection_name, batch_index):
        texts_for_embedding = []
        for item in batch:
            if item["type"] == "story":
                text_content = f"Title: {item.get('title', '')} {item.get('story_text', '') or ''}"
            elif item["type"] == "comment":
                text_content = f"Comment: {item.get('comment_text', '') or ''}"
            else:
                text_content = ""
            texts_for_embedding.append(text_content)

        if not texts_for_embedding:
            return

        try:
            embeddings = await embedding_model.aembed_documents(texts_for_embedding)
            points = [
                models.PointStruct(
                    id=item["id"],
                    vector=embedding,
                    payload=item
                )
                for item, embedding in zip(batch, embeddings)
            ]
            # If upsert is not async, wrap it in asyncio.to_thread
            upsert_fn = getattr(qdrant_client, 'upsert', None)
            if asyncio.iscoroutinefunction(upsert_fn):
                await qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
            else:
                await asyncio.to_thread(
                    qdrant_client.upsert,
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
            logger.info(f"Upserted {len(points)} points to Qdrant collection '{collection_name}' (batch {batch_index + 1})")
        except Exception as e:
            logger.error(f"Failed to process batch {batch_index + 1}: {e}")

    async def limited_process_batch(*args, **kwargs):
        async with semaphore:
            await process_batch(*args, **kwargs)

    tasks = [
        limited_process_batch(
            items_to_process[i:i+BATCH_SIZE],
            embedding_model,
            qdrant_client,
            collection_name,
            i // BATCH_SIZE
        )
        for i in range(0, len(items_to_process), BATCH_SIZE)
    ]

    await asyncio.gather(*tasks)

    return {} # No direct state update from this node, as data is in Qdrant


