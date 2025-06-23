import json
from typing import List
from .models import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from tenacity import retry, stop_after_attempt, wait_exponential
from .x_agent_config import logger, GOOGLE_API_KEY

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
async def generate_hashtags(keywords: List[str], expectation: str, goals: List[str], 
                          instructions: str, description: str) -> List[str]:
    """Generate strategic hashtags using LLM"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY)
        hashtag_prompt = PromptTemplate(
            input_variables=["keywords", "expectation", "goals", "instructions", "description"],
            template=(
                "You are a social media marketing expert. Generate strategic hashtags for a marketing campaign.\n\n"
                "CAMPAIGN DETAILS:\n"
                "Marketing Goals: {goals}\n"
                "Campaign Instructions: {instructions}\n"
                "Product Description: {description}\n"
                "Campaign Expectation: {expectation}\n"
                "Key Keywords: {keywords}\n\n"
                "HASHTAG REQUIREMENTS:\n"
                "1. Generate 8-12 strategic hashtags that:\n"
                "   - Align with the marketing goals\n"
                "   - Follow campaign instructions\n"
                "   - Highlight product benefits from description\n"
                "   - Match campaign expectations\n"
                "   - Target the specific audience\n"
                "   - Include industry trends\n"
                "   - Drive engagement\n\n"
                "2. Include a mix of:\n"
                "   - Brand hashtags (product/company specific)\n"
                "   - Industry hashtags (sector/trend related)\n"
                "   - Audience hashtags (target demographic)\n"
                "   - Campaign hashtags (goal-oriented)\n"
                "   - Product feature hashtags\n"
                "   - Benefit-focused hashtags\n\n"
                "3. Best Practices:\n"
                "   - Keep hashtags concise and memorable\n"
                "   - Use relevant industry terminology\n"
                "   - Mix popular and niche hashtags\n"
                "   - Ensure hashtags are searchable\n"
                "   - Align with campaign goals\n"
                "   - Reflect product benefits\n\n"
                "4. Return ONLY a valid JSON array of strings\n"
                "5. Each hashtag must start with #\n"
                "6. NO markdown formatting or code blocks\n\n"
                "Example format:\n"
                "[\"#BrandName\", \"#IndustryTrend\", \"#TargetAudience\", \"#CampaignGoal\", \"#ProductFeature\"]\n\n"
                "YOUR RESPONSE:"
            )
        )
        hashtag_chain = LLMChain(llm=llm, prompt=hashtag_prompt)

        raw_output = await hashtag_chain.arun(
            keywords=", ".join(keywords),
            expectation=expectation,
            goals=", ".join(goals),
            instructions=instructions,
            description=description or "Not provided"
        )
        logger.warning("Raw LLM output: %r", raw_output)
        
        try:
            expanded = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("LLM returned invalid JSON: %s", raw_output)
            raise e
        result = list(set(keywords + expanded))[:15]
        logger.info("Generated hashtags", hashtags=result)
        return result
    except Exception as e:
        logger.warning("Hashtag expansion failed", error=str(e))
        return keywords 



async def search_hashtags_node(state: AgentState) -> AgentState:
    """Generate hashtags for the Twitter agent"""
    if state.get("error"):
        return state
    
    try:
        hashtags = await generate_hashtags(
            keywords=state["company_keywords"],
            expectation=state["expectation"],
            goals=state["goals"],
            instructions=state["instructions"],
            description=state["description"]
        )
        state["hashtags"] = hashtags
        logger.info("Hashtags generated", agent_name=state["agent_name"], hashtags=hashtags)
    except Exception as e:
        state["error"] = f"Hashtag search failed: {str(e)}"
        logger.error("Hashtag search failed", agent_name=state["agent_name"], error=str(e))
    
    return state