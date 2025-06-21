import structlog
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from .models import AgentState, HNStory, HNComment, HNAgentOutput
from app.core.llm_client import get_llm
from .content_fetch import fetch_hn_story

logger = structlog.get_logger()

async def process_search_results_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to process and filter search results.
    """
    agent_input = state["agent_input"]
    search_results = state["search_results"]
    
    # Filter results based on criteria
    filtered_results = []
    for result in search_results:
        # Check minimum score
        if result.get("score", 0) < agent_input.min_score:
            continue
            
        # Check if story is within max age
        story_time = result.get("time", 0)
        if story_time > 0:  # Only filter if we have a valid timestamp
            from datetime import datetime, timedelta
            story_date = datetime.fromtimestamp(story_time)
            max_age_date = datetime.now() - timedelta(days=agent_input.max_age_days)
            if story_date < max_age_date:
                continue
        
        filtered_results.append(result)
    
    # Convert to HNStory objects
    hn_stories = []
    for result in filtered_results:
        story = HNStory(
            id=result.get("id"),
            title=result.get("title", ""),
            text=result.get("text"),
            url=result.get("url"),
            score=result.get("score", 0),
            time=result.get("time", 0),
            relevance=result.get("relevance_score", 0.0)
        )
        hn_stories.append(story)
    
    logger.info(f"Filtered {len(hn_stories)} stories from {len(search_results)} search results")
    return {"filtered_stories": hn_stories}

async def summarize_stories_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to generate summaries for stories using LLM.
    """
    agent_input = state["agent_input"]
    filtered_stories = state["filtered_stories"]
    
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["title", "text", "goals"],
        template=(
            "Summarize this Hacker News story in 2-3 sentences, focusing on aspects relevant to: {goals}\n\n"
            "Title: {title}\n"
            "Content: {text}\n\n"
            "Summary:"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    summarized_stories = []
    for story in filtered_stories:
        try:
            # Create content for summarization
            content = story.text or story.title
            if not content:
                continue
                
            summary = await chain.arun(
                title=story.title,
                text=content,
                goals=", ".join(agent_input.goals)
            )
            
            # Update story with summary
            story.summary = summary.strip()
            summarized_stories.append(story)
            
        except Exception as e:
            logger.error(f"Error summarizing story {story.id}: {e}")
            summarized_stories.append(story)  # Keep story without summary
    
    logger.info(f"Generated summaries for {len(summarized_stories)} stories")
    return {"summarized_stories": summarized_stories}

async def fetch_top_comments(story_id: int, limit=3) -> List[HNComment]:
    """Fetch top-level comments for a story."""
    story = await fetch_hn_story(story_id)
    if not story or not story.get("kids"):
        return []
    
    comments = []
    for comment_id in story["kids"][:limit]:
        comment_data = await fetch_hn_story(comment_id)
        if comment_data and comment_data.get("type") == "comment":
            comment = HNComment(
                id=comment_id,
                text=comment_data.get("text"),
                parent_story_id=story_id
            )
            comments.append(comment)
    
    return comments

async def curate_comments_and_final_output_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to fetch top comments and create final output.
    """
    agent_input = state["agent_input"]
    summarized_stories = state["summarized_stories"]
    
    # Fetch top comments for each story
    for story in summarized_stories:
        try:
            top_comments = await fetch_top_comments(story.id, 3)
            story.top_comments = top_comments
        except Exception as e:
            logger.error(f"Error fetching comments for story {story.id}: {e}")
            story.top_comments = []
    
    # Create final output
    output = HNAgentOutput(
        agent_name=agent_input.agent_name,
        goals=agent_input.goals,
        instructions=agent_input.instructions,
        stories=summarized_stories
    )
    
    logger.info(f"Final output created with {len(summarized_stories)} stories")
    return {"final_output": output} 