from ..state import AgentState
from ..reddit_client import get_reddit_client
from app.core.llm_client import get_llm
import structlog
import json
import asyncio

logger = structlog.get_logger()

async def search_subreddits_node(state: AgentState) -> AgentState:
    try:
        target_subreddits = {}
        async with await get_reddit_client() as reddit:
            keywords = state["keywords"]
            logger.info("Using keywords", agent_name=state["agent_name"], keywords=keywords)
            async def search_keyword(kw):
                result = {}
                try:
                    subreddits = reddit.subreddits.search(kw, limit=10)
                    async for subreddit in subreddits:
                        result[subreddit.display_name] = subreddit.public_description or subreddit.description
                except Exception as e:
                    logger.warning("Keyword search failed", keyword=kw, error=str(e))
                return result
            for i in range(0, len(keywords), 10):
                batch = keywords[i:i+10]
                results = await asyncio.gather(*(search_keyword(kw) for kw in batch))
                for res in results:
                    target_subreddits.update(res)
                logger.info("Batch processed", batch=batch)
                await asyncio.sleep(2)
            subreddit_data = [{"name": name, "description": description} for name, description in target_subreddits.items()]
            llm = get_llm()
            from langchain_core.prompts import PromptTemplate
            from langchain.chains import LLMChain
            prompt = PromptTemplate(
                input_variables=["subreddits", "expectation", "description", "target_audience"],
                template=(
                    "Evaluate these subreddits for relevance to our marketing needs. "
                    "Follow these instructions carefully:\n\n"
                    "PRODUCT DETAILS:\n"
                    "- Goals: {expectation}\n"
                    "- Description: {description}\n"
                    "- Target Audience: {target_audience}\n\n"
                    "SUBREDDITS TO EVALUATE:\n"
                    "{subreddits}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Select all relevant subreddits\n"
                    "2. Consider relevance, activity level, and audience match\n"
                    "3. Return ONLY a valid JSON array formatted exactly like this example:\n"
                    "   [\"subreddit1\", \"subreddit2\", \"subreddit3\"]\n"
                    "4. DO NOT include any backticks, markdown formatting, or extra explanation\n"
                    "5. DO NOT wrap the output in a code block\n"
                    "6. Ensure all subreddit names are from the provided list and spelled exactly\n\n"
                    "OUTPUT REQUIREMENTS:\n"
                    "- Must be a valid JSON array of strings\n"
                    "- Must use double quotes for each string\n"
                    "- Must NOT include backticks, markdown, or any extra text\n\n"
                    "IMPORTANT:\n"
                    "- NO markdown formatting (no ```json ... ```)"
                    "- NO text before or after the JSON array\n\n"
                    "YOUR RESPONSE:"
                )
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            BATCH_SIZE = 10
            final_subreddits = []
            for i in range(0, len(subreddit_data), BATCH_SIZE):
                batch = subreddit_data[i:i + BATCH_SIZE]
                batch_json = json.dumps(batch)
                try:
                    response = await chain.arun(
                        subreddits=batch_json,
                        expectation=state["expectation"],
                        description=state["description"],
                        target_audience=state["target_audience"]
                    )
                    logger.info("Response received", response=response)
                    batch_results = json.loads(response.lower())
                    final_subreddits.extend(batch_results)
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.warning(f"Batch processing failed: {str(e)}", batch_index=i//BATCH_SIZE, error=str(e))
                    continue
            logger.info("Relevant subreddits selected", agent_name=state["agent_name"], relevant_subreddits=final_subreddits)
            state["subreddits"] = final_subreddits
    except Exception as e:
        state["error"] = f"Subreddit search failed: {str(e)}"
        logger.error("Subreddit search failed", agent_name=state["agent_name"], error=str(e))
    return state 