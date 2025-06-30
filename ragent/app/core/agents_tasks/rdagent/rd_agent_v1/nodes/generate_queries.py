import json
import structlog
from app.core.llm_client import get_llm

logger = structlog.get_logger()

async def generate_queries_node(state):
    """
    1. Generates a comprehensive list of search queries for finding relevant Reddit discussions based on company information.
    2. This function takes the current state of the agent as input and generates a list of search queries that can be used to find relevant discussions on Reddit. 
    3. The queries are generated based on various aspects of the company, including its name, goals, keywords, description, target audience, and expected content. 
    4. The function returns the state with the generated queries.

    Parameters:
    - state (dict): The current state of the agent, containing information about the company and its goals.

    Returns:
    - state (dict): The updated state with the generated search queries.
    """

    logger.info("Starting generate_queries_node", node_name="generate_queries_node")
    
    if state.get("error"):
        return state
    try:
        llm = get_llm()
        # Get company data from state
        company_data = {
            "agent_name": state.get("agent_name", ""),
            "goals": state.get("goals", []),
            "instructions": state.get("instructions", ""),
            "keywords": state.get("keywords", []),  
            "description": state.get("description", ""),
            "target_audience": state.get("target_audience", ""),
            "expectation": state.get("expectation", "")
        }

        # Define the prompt

        prompt = f"""Based on the following company information, generate a comprehensive list of search queries for finding relevant Reddit discussions. 
        Focus on generating queries that will help find posts about our product, its broader category, target audience, and industry.
        Keep each query under 50 characters and make them specific and relevant.

        Company Information:
        - Name: {company_data['agent_name']}
        - Goals: {', '.join(company_data['goals'])}
        - Keywords: {', '.join(company_data['keywords'])}
        - Description: {company_data['description']}
        - Target Audience: {company_data['target_audience']}
        - Expected Content: {company_data['expectation']}

        Generate queries in the following categories:
        1. Core product terms (specific to our product)
        2. Product category terms (broader category our product belongs to)
        3. Feature-specific terms (both our product and similar products)
        4. Platform-specific terms
        5. Use case terms (both specific and general use cases)
        6. Target audience terms (including broader audience segments)
        7. Industry terms (both specific and general industry discussions)
        8. Alternative solutions (competitors and similar products)
        9. Problem space terms (common issues our product solves)
        10. Market trends and discussions

        For each category, include:
        - Specific queries about our product
        - Broader queries about the product category
        - Related industry discussions
        - Common pain points and solutions
        - Market trends and developments

        OUTPUT REQUIREMENTS:
        - Must be a valid JSON array of strings
        - Each string must be a search query
        - Each query must be under 50 characters
        - Must use double quotes for strings
        - Must NOT include backticks, markdown, or any extra text

        IMPORTANT:
        - NO markdown formatting (no ```json ... ```)
        - NO text before or after the JSON array
        - NO explanations or additional text

        YOUR RESPONSE:"""

        response = await llm.ainvoke(prompt)
        raw = response.content if hasattr(response, 'content') else response
        logger.info(f"Raw LLM query response: {raw}")

       # Cleaning the raw data

        cleaned = raw.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Parsing the cleaned data
        try:
            queries = json.loads(cleaned)
            if not isinstance(queries, list):
                raise ValueError("LLM did not return a list")
        except Exception as e:
            logger.error("Failed to parse LLM query response as JSON", error=str(e), response=raw)
            queries = company_data['keywords']
            
        # Keep company keywords also in the queries
        queries = list(set(queries + company_data['keywords']))
        queries = [q for q in queries if len(q) <= 50]
        state["generated_queries"] = queries
        logger.info("Generated search queries using LLM", agent_name=state.get("agent_name", ""), queries=queries)
        
    except Exception as e:
        state["error"] = f"Query generation failed: {str(e)}"
        logger.error("Query generation failed", error=str(e))
    return state 