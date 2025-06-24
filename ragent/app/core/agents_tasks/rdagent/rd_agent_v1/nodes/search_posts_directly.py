from ..state import AgentState
from ..reddit_client import get_reddit_client
from app.core.llm_client import get_llm, get_embedding_model
from app.core.vector_db import get_qdrant_client
import structlog
import json
from datetime import datetime
import asyncio
from qdrant_client.http.models import VectorParams, Distance, PointStruct

logger = structlog.get_logger()

async def safe_gemini_call(llm, prompt):
    try:
        response = await llm.ainvoke(prompt)
        return response.content.lower()
    except Exception as e:
        logger.warning(f"Gemini call failed: {str(e)}")
        raise

async def search_posts_directly_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    try:
        reddit = await get_reddit_client()
        llm = get_llm()
        qdrant_client = get_qdrant_client()
        embedding_model = get_embedding_model()
        collection_name = "reddit_posts"
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
        company_data = {
            "agent_name": state.get("agent_name", ""),
            "goals": state.get("goals", []),
            "instructions": state.get("instructions", ""),
            "company_keywords": state.get("company_keywords", []),
            "description": state.get("description", ""),
            "target_audience": state.get("target_audience", ""),
            "expectation": state.get("expectation", ""),
            "keywords": state.get("keywords","")
        }
        search_queries=company_data["keywords"]
        
        # prompt = f"""Based on the following company information, generate a comprehensive list of search queries for finding relevant Reddit discussions. 
        # Focus on generating queries that will help find posts about our product, its broader category, target audience, and industry.
        # Keep each query under 50 characters and make them specific and relevant.

        # Company Information:
        # - Name: {company_data['agent_name']}
        # - Goals: {', '.join(company_data['goals'])}
        # - Keywords: {', '.join(company_data['company_keywords'])}
        # - Description: {company_data['description']}
        # - Target Audience: {company_data['target_audience']}
        # - Expected Content: {company_data['expectation']}

        # Generate queries in the following categories:
        # 1. Core product terms (specific to our product)
        # 2. Product category terms (broader category our product belongs to)
        # 3. Feature-specific terms (both our product and similar products)
        # 4. Platform-specific terms
        # 5. Use case terms (both specific and general use cases)
        # 6. Target audience terms (including broader audience segments)
        # 7. Industry terms (both specific and general industry discussions)
        # 8. Alternative solutions (competitors and similar products)
        # 9. Problem space terms (common issues our product solves)
        # 10. Market trends and discussions

        # For each category, include:
        # - Specific queries about our product
        # - Broader queries about the product category
        # - Related industry discussions
        # - Common pain points and solutions
        # - Market trends and developments

        # OUTPUT REQUIREMENTS:
        # - Must be a valid JSON array of strings
        # - Each string must be a search query
        # - Each query must be under 50 characters
        # - Must use double quotes for strings
        # - Must NOT include backticks, markdown, or any extra text

        # IMPORTANT:
        # - NO markdown formatting (no ```json ... ```)
        # - NO text before or after the JSON array
        # - NO explanations or additional text

        # YOUR RESPONSE:"""

        # gemini_response = await safe_gemini_call(llm, prompt)
        # try:
        #     cleaned_response = gemini_response.strip()
        #     if cleaned_response.startswith("```json"):
        #         cleaned_response = cleaned_response[7:]
        #     if cleaned_response.endswith("```"):
        #         cleaned_response = cleaned_response[:-3]
        #     cleaned_response = cleaned_response.strip()
        #     search_queries = json.loads(cleaned_response)
        #     if not isinstance(search_queries, list):
        #         raise ValueError("Invalid response format")
        # except (json.JSONDecodeError, ValueError) as e:
        #     logger.error("Failed to parse Gemini response", error=str(e), response=gemini_response)
            # search_queries = company_data['company_keywords']

        # search_queries = list(set(search_queries + company_data['company_keywords']))
        # search_queries = [q for q in search_queries if len(q) <= 50]
        # state["keywords"] = search_queries
        # logger.info("Generated search queries using Gemini", agent_name=state["agent_name"], queries=search_queries)
        BATCH_SIZE = 50
        EMBEDDING_BATCH_SIZE = 10
        current_batch = []
        posts = []
        seen_post_ids = set()
        async def process_submission_batch(submissions, current_query):
            tasks = []
            for submission in submissions:
                if submission.id in seen_post_ids:
                    continue
                seen_post_ids.add(submission.id)
                content = (submission.title + " " + submission.selftext).lower()
                keyword_matches = sum(1 for kw in search_queries if kw.lower() in content)
                keyword_relevance = min(1.0, keyword_matches / max(1, len(state["company_keywords"])))
                post_data = {
                    "subreddit": submission.subreddit.display_name,
                    "post_id": submission.id,
                    "post_title": submission.title,
                    "post_body": submission.selftext,
                    "post_url": f"https://www.reddit.com{submission.permalink}",
                    "upvotes": submission.score,
                    "comment_count": submission.num_comments,
                    "created": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                    "keyword_relevance": keyword_relevance,
                    "matched_query": current_query,
                    "text": f"{submission.title} {submission.selftext}"
                }
                tasks.append(post_data)
            return tasks
        async def generate_embeddings_batch(texts):
            try:
                embeddings = await embedding_model.aembed_documents(texts)
                return embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                return [None] * len(texts)
        async with reddit as reddit_client:
            all_subreddit = await reddit_client.subreddit("all")
            async def process_query(query):
                submissions = []
                try:
                    async for submission in all_subreddit.search(
                        query=query,
                        sort="relevance",
                        time_filter="month",
                        limit=30
                    ):
                        submissions.append(submission)
                        if len(submissions) >= 30:
                            break
                except Exception as e:
                    logger.warning(f"Search failed for query: {query}", error=str(e))
                return submissions, query
            query_tasks = [process_query(query) for query in search_queries]
            query_results = await asyncio.gather(*query_tasks)
            for submissions, current_query in query_results:
                if not submissions:
                    continue
                batch_posts = await process_submission_batch(submissions, current_query)
                if not batch_posts:
                    continue
                for j in range(0, len(batch_posts), EMBEDDING_BATCH_SIZE):
                    embedding_batch = batch_posts[j:j + EMBEDDING_BATCH_SIZE]
                    texts = [post["text"] for post in embedding_batch]
                    embeddings = await generate_embeddings_batch(texts)
                    for post, embedding in zip(embedding_batch, embeddings):
                        if embedding is None:
                            continue
                        current_batch.append(PointStruct(
                            id=abs(hash(post["post_id"])) % (2**63),
                            vector=embedding,
                            payload={
                                "submission_id": post["post_id"],
                                "title": post["post_title"],
                                "content": post["post_body"],
                                "url": post["post_url"],
                                "created_utc": datetime.fromisoformat(post["created"]).timestamp(),
                                "subreddit": post["subreddit"],
                                "score": post["upvotes"],
                                "num_comments": post["comment_count"],
                                "keyword_relevance": post["keyword_relevance"]
                            }
                        ))
                        post.pop("text", None)
                        posts.append(post)
                    if len(current_batch) >= BATCH_SIZE:
                        try:
                            qdrant_client.upsert(
                                collection_name=collection_name,
                                points=current_batch
                            )
                            current_batch = []
                        except Exception as e:
                            logger.error(f"Error upserting batch: {str(e)}")
                            current_batch = []
                if len(posts) >= 500:
                    break
            if current_batch:
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=current_batch
                    )
                except Exception as e:
                    logger.error(f"Error upserting final batch: {str(e)}")
        try:
            query_text = f"""
            {company_data['description']}
            {', '.join(company_data['goals'])}
            {', '.join(search_queries)}
            {company_data['target_audience']}
            {company_data['expectation']}
            """
            query_embedding = await embedding_model.aembed_query(query_text)
            semantic_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=100,
                score_threshold=0.5
            )
            post_dict = {post["post_id"]: post for post in posts}
            semantic_posts = []
            for result in semantic_results:
                post_id = result.payload["submission_id"]
                if post_id in post_dict:
                    post = post_dict[post_id]
                    post["semantic_relevance"] = result.score
                    post["combined_relevance"] = (
                        0.8 * result.score + 
                        0.2 * post["keyword_relevance"]
                    )
                    semantic_posts.append(post)
            semantic_posts.sort(key=lambda x: x["combined_relevance"], reverse=True)
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            posts.sort(key=lambda x: (x["keyword_relevance"], x["upvotes"]), reverse=True)
        state["posts"] = semantic_posts
        logger.info("Direct post search completed with semantic ranking", agent_name=state["agent_name"], posts_found=len(state["posts"]), unique_queries_used=search_queries)
    except Exception as e:
        state["error"] = f"Direct post search failed: {str(e)}"
        logger.error("Direct post search failed", error=str(e))
    return state 