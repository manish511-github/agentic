import structlog
from typing import Dict, Any, List
from qdrant_client.http import models
from .models import AgentState
from app.core.vector_db import get_qdrant_client, get_collection_name
from app.core.llm_client import get_embedding_model

logger = structlog.get_logger()




async def semantic_search_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to perform semantic search using the vector database.
    """
    agent_input = state["agent_input"]
    expanded_queries = state["expanded_queries"]
    
    # Get collection name for this agent
    collection_name = get_collection_name(agent_input.agent_name, "hackernews")
    
    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()
    
    query_text = f"""
    Context for search: {agent_input.description or ''}.
    Goals: {', '.join(agent_input.goals)}.
    Company Keywords: {', '.join(agent_input.company_keywords)}.
    Target Audience: {agent_input.target_audience or ''}.
    Expectation: {agent_input.expectation}.
    """

    query_embedding = await embedding_model.aembed_query(query_text)
    
    # now_unix_timestamp = datetime.utcnow().timestamp()
    # min_time_unix_timestamp = now_unix_timestamp - agent_input.max_age_days * 86400
    # # Perform semantic search for each expanded query
    # search_filter = models.Filter(
    #     must=[
    #         models.FieldCondition(
    #             key="time",
    #             range=models.Range(gte=min_time_unix_timestamp)
    #         ),
    #         models.FieldCondition(
    #             key="score",
    #             range=models.Range(gte=agent_input.min_score)
    #         )
    #     ]
    # )
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=50, # Get more results to find relevant comments
            score_threshold=0.5,
            with_payload=True,
            with_vectors=False
        )
        logger.info(f"Performed semantic search in '{collection_name}', found {len(results)} relevant items.")
        return {"raw_search_results": results}
    except Exception as e:
        logger.error(f"Semantic search failed in collection '{collection_name}': {e}")
        return {"raw_search_results": []}