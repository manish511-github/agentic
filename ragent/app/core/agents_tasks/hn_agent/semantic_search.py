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
    search_results = []
    
    # Perform semantic search for each expanded query
    for query in expanded_queries:
        try:
            # Generate embedding for the query
            query_embedding = await embedding_model.aembed_query(query)
            
            # Search in Qdrant
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=20,
                with_payload=True,
                with_vectors=False
            )
            
            # Add query context to results
            for result in search_result:
                result.payload["query"] = query
                result.payload["relevance_score"] = result.score
                search_results.append(result.payload)
                
        except Exception as e:
            logger.error(f"Error performing semantic search for query '{query}' in collection '{collection_name}': {e}")
    
    # Remove duplicates and sort by relevance
    unique_results = {}
    for result in search_results:
        story_id = result.get("id")
        if story_id not in unique_results or result.get("relevance_score", 0) > unique_results[story_id].get("relevance_score", 0):
            unique_results[story_id] = result
    
    # Sort by relevance score (higher is better)
    sorted_results = sorted(
        unique_results.values(),
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )
    
    # Take top results
    top_results = sorted_results[:50]  # Limit to top 50 most relevant
    
    logger.info(f"Found {len(top_results)} relevant stories from semantic search in collection '{collection_name}'")
    return {"search_results": top_results} 