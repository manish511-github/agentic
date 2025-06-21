import asyncio
import structlog
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from .models import AgentState, HNStory, HNComment, HNAgentOutput
from app.core.llm_client import get_llm
from .content_fetch import fetch_hn_story

logger = structlog.get_logger()

BATCH_SIZE = 10  # or whatever batch size you prefer

async def process_search_results_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to process raw search results, separate stories and comments,
    and prepare them for further processing.
    """
    raw_search_results = state["raw_search_results"]
    found_stories_map: Dict[int, HNStory] = {}
    relevant_comments_found: List[HNComment] = []

    for result in raw_search_results:
        payload = result.payload
        item_id = result.id
        item_type = payload.get("type")

        if item_type == "story":
            if item_id not in found_stories_map:
                story_obj = HNStory(
                    id=item_id,
                    title=payload.get("title", ""),
                    text=payload.get("story_text"),
                    url=payload.get("url"),
                    score=payload.get("score", 0),
                    time=payload.get("time", 0),
                    relevance=result.score,
                    summary=None,
                    top_comments=[],
                    story_id=payload.get("story_id"),
                    children=payload.get("children", []),
                    relevant_comment_ids=[]
                )
                found_stories_map[item_id] = story_obj
        elif item_type == "comment":
            if payload.get("comment_text"):
                comment_obj = HNComment(
                    id=item_id,
                    text=payload["comment_text"],
                    relevance=result.score,
                    parent_story_id=payload.get("parent_story_id"),
                    story_id=payload.get("story_id"),
                    children=payload.get("children", []),
                    relevant_comment_ids=[]
                )
                relevant_comments_found.append(comment_obj)

    logger.info(f"Identified {len(found_stories_map)} relevant stories and {len(relevant_comments_found)} relevant comments from search results.")
    return {
        "found_stories_map": found_stories_map,
        "relevant_comments_found": relevant_comments_found
    }

async def summarize_stories_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to concurrently summarize identified stories.
    """
    pass
    # found_stories_map = state["found_stories_map"]
    # llm = get_llm()

    # async def _summarize_single_story(story_payload: Dict) -> Optional[str]:
    #     story_title = story_payload.get("title", "")
    #     story_text = story_payload.get("text", "") or ""

    #     if not story_title and not story_text:
    #         return None

    #     prompt = f"Summarize the following Hacker News content in 2 concise sentences. Focus on the main topic and key takeaway:\nTitle: {story_title}\nContent: {story_text}"
    #     try:
    #         response = await llm.ainvoke(prompt)
    #         return response.content.strip()
    #     except Exception as e:
    #         logger.error(f"Failed to summarize story {story_payload.get('id')}: {e}")
    #         return None

    # summarization_tasks = []
    # for story_id, story_obj in found_stories_map.items():
    #     summarization_tasks.append(
    #         asyncio.create_task(_summarize_single_story(story_obj.model_dump()))
    #     )
    # summaries = await asyncio.gather(*summarization_tasks)

    # # Assign summaries back to the stories in the map
    # for i, story_id in enumerate(found_stories_map.keys()):
    #     found_stories_map[story_id].summary = summaries[i]

    # logger.info(f"Summarized {len(summaries)} stories.")
    # return {"found_stories_map": found_stories_map} # Update the map with summaries

async def curate_comments_and_final_output_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to link relevant comments to stories, sort them, and prepare final output.
    """
    found_stories_map = state["found_stories_map"]
    relevant_comments_found = state["relevant_comments_found"]

    placeholder_story_ids = set()
    for comment in relevant_comments_found:
        if comment.story_id in found_stories_map:
            story = found_stories_map[comment.story_id]
            if not hasattr(story, "relevant_comment_ids"):
                story.relevant_comment_ids = []
            story.relevant_comment_ids.append(comment.id)
        else:
            found_stories_map[comment.story_id] = HNStory(
                id=comment.story_id,
                title="[Unknown Story]",
                text=None,
                url=None,
                score=0,
                time=0,
                relevance=comment.relevance or 0.0,
                summary=None,
                top_comments=[comment],
                story_id=comment.story_id,
                children=[],
                relevant_comment_ids=[comment.id]
            )
            placeholder_story_ids.add(comment.story_id)

    # Fetch real story data for placeholders and update the mapping
    if placeholder_story_ids:
        fetched_stories = asyncio.get_event_loop().run_until_complete(
            fetch_stories_in_batches(placeholder_story_ids, fetch_hn_story, BATCH_SIZE)
        )
        for story_data in fetched_stories:
            if story_data and story_data.get("id") in found_stories_map:
                story_obj = found_stories_map[story_data["id"]]
                story_obj.title = story_data.get("title", "[Unknown Story]")
                story_obj.text = story_data.get("story_text")
                story_obj.url = story_data.get("url")
                story_obj.score = story_data.get("score", 0)
                story_obj.time = story_data.get("time", 0)
                # You can update more fields as needed

    # final_stories_output = sorted(
    #     list(found_stories_map.values()),
    #     key=lambda s: s.relevance,
    #     reverse=True
    # )
    final_stories_output=list(found_stories_map.values())

    logger.info(f"Curated comments and prepared final output with {len(final_stories_output)} stories.")
    return {"final_stories_output": final_stories_output}

async def fetch_stories_in_batches(story_ids, fetch_fn, batch_size=BATCH_SIZE):
    all_results = []
    story_ids = list(story_ids)
    for i in range(0, len(story_ids), batch_size):
        batch = story_ids[i:i+batch_size]
        fetch_tasks = [fetch_fn(story_id) for story_id in batch]
        results = await asyncio.gather(*fetch_tasks)
        all_results.extend(results)
    return all_results