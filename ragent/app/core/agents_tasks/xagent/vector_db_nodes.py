import asyncio
import uuid
from app.core.vector_db import get_qdrant_client, get_collection_name
from app.core.llm_client import get_embedding_model
from qdrant_client.http import models
from fastapi import HTTPException
import structlog

logger = structlog.get_logger()

async def store_tweets_in_vector_db_node(state):
    """
    Node to batch, embed, and upsert tweets into the Qdrant vector database.
    Deletes and recreates the collection for the agent each run.
    """
    agent_name = state["agent_name"]
    tweets = state["tweets"]
    collection_name = get_collection_name(agent_name, "tweets")
    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()

    # Delete and recreate the collection (reset for each run)
    try:
        try:
            qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted existing Qdrant collection: {collection_name}")
        except Exception as e:
            logger.info(f"Qdrant collection '{collection_name}' did not exist or could not be deleted: {e}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
        logger.info(f"Created new Qdrant collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create/configure Qdrant collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set up Qdrant collection: {str(e)}")

    # Batch, embed, and upsert tweets
    BATCH_SIZE = 500
    CONCURRENCY_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    items_to_process = tweets

    async def process_batch(batch, embedding_model, qdrant_client, collection_name, batch_index):
        texts_for_embedding = [tweet["text"] for tweet in batch]
        if not texts_for_embedding:
            return
        try:
            embeddings = await embedding_model.aembed_documents(texts_for_embedding)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=tweet
                )
                for tweet, embedding in zip(batch, embeddings)
            ]
            await asyncio.to_thread(
                qdrant_client.upsert,
                collection_name=collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Upserted {len(points)} tweets to Qdrant collection '{collection_name}' (batch {batch_index + 1})")
        except Exception as e:
            logger.error(f"Failed to process tweet batch {batch_index + 1}: {e}")

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
    return state

async def semantic_search_tweets_node(state):
    """
    Node to perform semantic search over tweets in the Qdrant vector database.
    Expects 'agent_name' and 'semantic_query' in state.
    Results are stored in state['semantic_results'].
    """
    agent_name = state["agent_name"]
    query = f"""
    Context for search: {state['description'] or ''}.
    Goals: {', '.join(state['goals'])}.
    Company Keywords: {', '.join(state['company_keywords'])}.
    Target Audience: {state['target_audience'] or ''}.
    Expectation: {state['expectation']}.
    """
    # if not query:
    #     logger.error("No semantic_query provided in state for semantic search.")
    #     state["semantic_results"] = []
    #     return state
    collection_name = get_collection_name(agent_name, "tweets")
    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()
    query_embedding = await embedding_model.aembed_query(query)
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20,
            score_threshold=0.5,
            with_payload=True,
            with_vectors=False
        )
        logger.info(f"Performed semantic search in '{collection_name}', found {len(results)} relevant tweets.")
        state["semantic_results"] = results
        return state
    except Exception as e:
        logger.error(f"Semantic search failed in collection '{collection_name}': {e}")
        state["semantic_results"] = []
        return state 