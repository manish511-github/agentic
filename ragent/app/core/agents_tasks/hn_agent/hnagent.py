import structlog
from fastapi import APIRouter, HTTPException
from langgraph.graph import StateGraph, END
from typing import Any
from .models import HNAgentInput, HNAgentOutput, AgentState
from .query_expansion import expand_queries_node
from .content_fetch import fetch_and_store_content_node
from .semantic_search import semantic_search_node
from .result_processing import (
    process_search_results_node,
    summarize_stories_node,
    curate_comments_and_final_output_node
)
from app.core.vector_db import ensure_collection_exists

logger = structlog.get_logger()
router = APIRouter()

# Create the LangGraph workflow
def create_hn_workflow():
    """Create the Hacker News agent workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("expand_queries", expand_queries_node)
    workflow.add_node("fetch_and_store_content", fetch_and_store_content_node)
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("process_search_results", process_search_results_node)
    workflow.add_node("summarize_stories", summarize_stories_node)
    workflow.add_node("curate_comments_and_final_output", curate_comments_and_final_output_node)
    
    # Define the flow
    workflow.set_entry_point("expand_queries")
    workflow.add_edge("expand_queries", "fetch_and_store_content")
    workflow.add_edge("fetch_and_store_content", "semantic_search")
    workflow.add_edge("semantic_search", "process_search_results")
    workflow.add_edge("process_search_results", "summarize_stories")
    workflow.add_edge("summarize_stories", "curate_comments_and_final_output")
    workflow.add_edge("curate_comments_and_final_output", END)
    
    return workflow.compile()

# Create workflow instance
hn_workflow = create_hn_workflow()

@router.post("/hackernews/hn-agent", response_model=HNAgentOutput)
async def hn_agent(input_data: HNAgentInput) -> HNAgentOutput:
    """
    Hacker News Agent endpoint that finds relevant stories using semantic search.
    """
    try:
        # Ensure Qdrant collection exists
        ensure_collection_exists(input_data.agent_name, "hackernews")
        
        # Initialize state
        initial_state = {
            "agent_input": input_data,
            "expanded_queries": [],
            "stories_stored": 0,
            "search_results": [],
            "filtered_stories": [],
            "summarized_stories": [],
            "final_stories_output": None
        }
        
        # Run the workflow
        logger.info(f"Starting Hacker News agent workflow for: {input_data.agent_name}")
        result = await hn_workflow.ainvoke(initial_state)
        
        # Extract final output
        final_output = result.get("final_stories_output")
        if not final_output:
            raise HTTPException(status_code=500, detail="Failed to generate output")
        
        logger.info(f"Hacker News agent completed successfully for: {input_data.agent_name}")
        return {
            "agent_name": input_data.agent_name,
            "goals": input_data.goals,
            "instructions": input_data.instructions,
            "stories": final_output
        }
        
    except Exception as e:
        logger.error(f"Hacker News agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hacker News agent failed: {str(e)}")