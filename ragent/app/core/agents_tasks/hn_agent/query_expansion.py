import json
import structlog
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from .models import AgentState
from app.core.llm_client import get_llm

logger = structlog.get_logger()

async def expand_queries_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node to expand user goals and keywords into more detailed search queries for Hacker News.
    """
    agent_input = state["agent_input"]
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=[
            "goals",
            "keywords",
            "expectation",
            "description",
            "target_audience"
        ],
        template="""
            Based on the following company information, generate a comprehensive list of search queries for finding relevant Hacker News discussions.
            Focus on generating queries that will help find posts about our product, its broader category, target audience, and industry.
            Keep each query under 50 characters and make them specific and relevant.

            Company Information:
            - Goals: {goals}
            - Keywords: {keywords}
            - Description: {description}
            - Target Audience: {target_audience}
            - Expected Content: {expectation}

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
            - NO markdown formatting (no ```json ... ``` )
            - NO text before or after the JSON array
            - NO explanations or additional text

            YOUR RESPONSE:
            """
        )
    chain = LLMChain(llm=llm, prompt=prompt)
    raw_output = None
    try:
        raw_output = await chain.arun(
            goals=", ".join(agent_input.goals),
            keywords=", ".join(agent_input.company_keywords),
            expectation=agent_input.expectation,
            description=agent_input.description,
            target_audience=", ".join(agent_input.target_audience)
        )
        expanded_queries = json.loads(raw_output)
        if not isinstance(expanded_queries, list):
            raise ValueError("LLM did not return a JSON list.")
        logger.info(f"Expanded queries for Hacker News: {expanded_queries}")
        return {"expanded_queries": expanded_queries}
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"LLM query expansion for Hacker News failed to parse JSON, falling back to keywords only. Error: {e}, Raw output: {raw_output}"
        )
        return {"expanded_queries": agent_input.company_keywords}