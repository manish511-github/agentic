from ..state import RedditAgentState
from ..reddit_client import get_reddit_client
from app.core.llm_client import get_llm
import structlog
import json
import asyncio

# New shared configuration and rate limiting utilities
from ...settings import settings
from ...utils.rate_limiter import reddit_limiter

logger = structlog.get_logger()


async def search_subreddits_node(state: RedditAgentState) -> RedditAgentState:
    try:
        target_subreddits = {}
        reddit = await get_reddit_client()
        keywords = state["generated_queries"]
        logger.info("Using keywords",
                    agent_name=state["agent_name"], keywords=keywords)

        async def search_query_in_subreddit(query):
            result = {}
            try:
                # Honour global Reddit rate-limit
                async with reddit_limiter:
                    subreddits = reddit.subreddits.search(
                        query, limit=settings.posts_per_search)
                    async for subreddit in subreddits:
                        result[subreddit.display_name] = subreddit.public_description or subreddit.description
            except Exception as e:
                logger.warning("Keyword search failed",
                               keyword=query, error=str(e))
            return result
        
        for i in range(0, len(keywords), settings.embedding_batch_size):
            batch = keywords[i:i+settings.embedding_batch_size]
            results = await asyncio.gather(*(search_query_in_subreddit(query) for query in batch))
            logger.info("Results for searching subreddits for keywords: {batch}", batch=batch, results=len(results))
            for res in results:
                target_subreddits.update(res)
            logger.info("Batch processed", batch=batch)
        
        subreddit_data = [{"name": name, "description": description}
                          for name, description in target_subreddits.items()]
        
        llm = get_llm()
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
        prompt = PromptTemplate(
            input_variables=["subreddits", "expectation",
                             "description", "target_audience", "keywords", "generated_queries"],
            template=(
                "Evaluate these subreddits for relevance to our marketing needs. "
                "Follow these instructions carefully:\n\n"
                "PRODUCT DETAILS:\n"
                "- Goals: {expectation}\n"
                "- Description: {description}\n"
                "- Target Audience: {target_audience}\n\n"
                "- Keywords: {keywords}\n"
                "- Generated Queries: {generated_queries}\n\n"
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
        BATCH_SIZE = settings.llm_batch_size
        final_subreddits = []
        semaphore = asyncio.Semaphore(10)

        async def process_batch(batch, batch_index):
            batch_json = json.dumps(batch)
            async with semaphore:
                try:
                    response = await chain.arun(
                        subreddits=batch_json,
                        expectation=state["expectation"],
                        description=state["description"],
                        target_audience=state["target_audience"],
                        keywords=state["keywords"],
                        generated_queries=state["generated_queries"]
                    )
                    logger.info("Response received", response=response)
                    batch_results = json.loads(response.lower())
                    return batch_results
                except Exception as e:
                    logger.warning(
                        f"Batch processing failed: {str(e)}", batch_index=batch_index, error=str(e))
                    return []

        batches = [subreddit_data[i:i + BATCH_SIZE] for i in range(0, len(subreddit_data), BATCH_SIZE)]
        tasks = [process_batch(batch, idx) for idx, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)
        for batch_results in results:
            final_subreddits.extend(batch_results)
        
        logger.info("Relevant subreddits selected",
                    agent_name=state["agent_name"], relevant_subreddits=final_subreddits)
        state["subreddits"] = final_subreddits
        
    except Exception as e:
        state["error"] = f"Subreddit search failed: {str(e)}"
        logger.error("Subreddit search failed",
                     agent_name=state["agent_name"], error=str(e))
    
    return state
