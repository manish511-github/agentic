from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import structlog
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import ProjectModel

# Initialize logging
logger = structlog.get_logger()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# FastAPI router
router = APIRouter()

class AgentGeneratorInput(BaseModel):
    agent_name: str = Field(..., min_length=1, description="Name of the marketing agent")
    goals: List[str] = Field(
        ...,
        description="Goals for the agent (e.g., lead_generation, brand_awareness, engagement, support)",
        examples=[["lead_generation", "brand_awareness"]]
    )
    project_id: str = Field(..., description="UUID of the project")
    existing_context: Optional[str] = Field(
        None,
        description="Any existing personality or instructions that should be enhanced"
    )

class ExpectedOutcomesInput(AgentGeneratorInput):
    instructions: str = Field(..., description="The agent's instructions and personality profile")

class AgentGeneratorOutput(BaseModel):
    agent_name: str
    goals: List[str]
    context: str

class ExpectedOutcomesOutput(BaseModel):
    agent_name: str
    goals: List[str]
    expected_outcomes: str

GOAL_MAPPING = {
    "lead_generation": {
        "personality_traits": [
            "Solution-oriented",
            "Professional",
            "Value-focused",
            "Results-driven",
            "Customer-centric"
        ],
        "instruction_focus": [
            "Identify potential customers",
            "Highlight product benefits",
            "Create compelling calls-to-action",
            "Focus on value proposition",
            "Track conversion metrics"
        ]
    },
    "brand_awareness": {
        "personality_traits": [
            "Engaging",
            "Memorable",
            "Authentic",
            "Storytelling",
            "Brand-focused"
        ],
        "instruction_focus": [
            "Share brand stories",
            "Create memorable content",
            "Build brand recognition",
            "Maintain consistent messaging",
            "Engage with brand advocates"
        ]
    },
    "engagement": {
        "personality_traits": [
            "Conversational",
            "Friendly",
            "Community-focused",
            "Interactive",
            "Supportive"
        ],
        "instruction_focus": [
            "Foster community discussions",
            "Respond to user comments",
            "Create interactive content",
            "Build relationships",
            "Encourage participation"
        ]
    },
    "support": {
        "personality_traits": [
            "Helpful",
            "Empathetic",
            "Patient",
            "Knowledgeable",
            "Solution-focused"
        ],
        "instruction_focus": [
            "Address user concerns",
            "Provide clear solutions",
            "Share helpful resources",
            "Maintain positive tone",
            "Follow up on issues"
        ]
    }
}

@router.post("/generate-instruction-personality", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def generate_agent_profile(input: AgentGeneratorInput, db: AsyncSession = Depends(get_db)) -> AgentGeneratorOutput:
    try:
        # Fetch project details from database using async query
        stmt = select(ProjectModel).where(ProjectModel.uuid == input.project_id)
        result = await db.execute(stmt)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            max_retries=2,
            temperature=0.7
        )

        # Prepare goal-specific context
        goal_contexts = []
        for goal in input.goals:
            if goal.lower() in GOAL_MAPPING:
                context = GOAL_MAPPING[goal.lower()]
                goal_contexts.append(
                    f"Goal: {goal}\n"
                    f"Personality Traits: {', '.join(context['personality_traits'])}\n"
                    f"Instruction Focus: {', '.join(context['instruction_focus'])}"
                )

        # Create the prompt
        prompt = PromptTemplate(
            input_variables=["agent_name", "goals", "description", "target_audience", "company_keywords", "existing_context", "goal_contexts"],
            template=(
                "You are an expert marketing strategist. Create a focused agent profile for '{agent_name}', specifically for the product described below.\n\n"
                "PRODUCT DETAILS:\n"
                "Description: {description}\n"
                "Target Audience: {target_audience}\n"
                "Keywords: {company_keywords}\n"
                "Existing Context: {existing_context}\n\n"
                "GOAL CONTEXTS:\n{goal_contexts}\n\n"
                "REQUIREMENTS:\n"
                "- Make the personality and instructions specific to the product and its unique features\n"
                "- Keep the response concise and focused\n"
                "- Avoid unnecessary details and repetition\n"
                "- Focus on key personality traits and core instructions\n"
                "- Use clear, direct language\n"
                "- Maximum 2-3 sentences per key point\n"
                "- Use third-person perspective (avoid 'I am', 'I will')\n"
                "- Use present tense (avoid 'will', 'going to')\n"
                "- Use active voice\n"
                "- Maintain professional tone\n"
                "- Focus on actions and capabilities\n\n"
                "OUTPUT REQUIREMENTS:\n"
                "- Must be a valid JSON object\n"
                "- Must contain a single key-value pair with key 'context'\n"
                "- Value of 'context' must be a single paragraph combining personality and instructions\n"
                "- Must use double quotes for the key and value\n"
                "- Must NOT include backticks, markdown, or any extra text\n\n"
                "IMPORTANT:\n"
                "- NO markdown formatting (no ```json ... ```)\n"
                "- NO text before or after the JSON object\n\n"
                "YOUR RESPONSE:"
            )
        )


        # Create and run the chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        
        result = await chain.arun(
            agent_name=input.agent_name,
            goals=", ".join(input.goals),
            description=project.description or "Not specified",
            target_audience=project.target_audience or "Not specified",
            company_keywords=", ".join(project.keywords) if project.keywords else "Not specified",
            existing_context=input.existing_context or "Not specified",
            goal_contexts="\n".join(goal_contexts)
        )
        logger.info("result"+str(result))
        # Parse the result
        try:
            # Clean the response by removing markdown formatting
            cleaned_response = result.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            generated_profile = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response", error=str(e), response=result)
            raise HTTPException(status_code=500, detail="Failed to generate agent profile")

        # Create the output
        return AgentGeneratorOutput(
            agent_name=input.agent_name,
            goals=input.goals,
            context=generated_profile["context"]
        )

    except Exception as e:
        logger.error("Agent profile generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error generating agent profile: {str(e)}")

@router.post("/generate-expected-outcomes", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def generate_expected_outcomes(input: ExpectedOutcomesInput, db: AsyncSession = Depends(get_db)) -> ExpectedOutcomesOutput:
    try:
        # Fetch project details from database using async query
        stmt = select(ProjectModel).where(ProjectModel.uuid == input.project_id)
        result = await db.execute(stmt)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            max_retries=2,
            temperature=0.7
        )

        # Prepare goal-specific context
        goal_contexts = []
        for goal in input.goals:
            if goal.lower() in GOAL_MAPPING:
                context = GOAL_MAPPING[goal.lower()]
                goal_contexts.append(
                    f"Goal: {goal}\n"
                    f"Personality Traits: {', '.join(context['personality_traits'])}\n"
                    f"Instruction Focus: {', '.join(context['instruction_focus'])}"
                )

        # Create the prompt for expected outcomes
        prompt = PromptTemplate(
            input_variables=["agent_name", "goals", "description", "target_audience", "company_keywords", "instructions", "goal_contexts"],
            template=(
                "You are an expert marketing strategist. Create expected outcomes for '{agent_name}', specifically for the product described below.\n\n"
                "PRODUCT DETAILS:\n"
                "Description: {description}\n"
                "Target Audience: {target_audience}\n"
                "Keywords: {company_keywords}\n"
                "Instructions: {instructions}\n\n"
                "GOAL CONTEXTS:\n{goal_contexts}\n\n"
                "REQUIREMENTS:\n"
                "- Make the expected outcomes specific to the product and its unique value proposition\n"
                "- Keep the response concise and focused\n"
                "- Focus on qualitative outcomes and success indicators\n"
                "- Avoid specific percentage targets or numerical metrics\n"
                "- Describe observable changes and improvements\n"
                "- Align outcomes with agent's goals and instructions\n"
                "- Use clear, direct language\n"
                "- Maximum 2-3 sentences per key point\n"
                "- Use present tense\n"
                "- Use active voice\n"
                "- Maintain professional tone\n\n"
                "OUTPUT REQUIREMENTS:\n"
                "- Must be a valid JSON object\n"
                "- Must contain a single key-value pair with key 'expected_outcomes'\n"
                "- Value of 'expected_outcomes' must be a single paragraph\n"
                "- Must use double quotes for the key and value\n"
                "- Must NOT include backticks, markdown, or any extra text\n\n"
                "IMPORTANT:\n"
                "- NO markdown formatting (no ```json ... ```)\n"
                "- NO text before or after the JSON object\n"
                "- NO specific percentage targets\n\n"
                "YOUR RESPONSE:"
            )
        )

        # Create and run the chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        result = await chain.arun(
            agent_name=input.agent_name,
            goals=", ".join(input.goals),
            description=project.description or "Not specified",
            target_audience=project.target_audience or "Not specified",
            company_keywords=", ".join(project.keywords) if project.keywords else "Not specified",
            instructions=input.instructions,
            goal_contexts="\n".join(goal_contexts)
        )

        # Clean and parse the response
        try:
            cleaned_response = result.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            generated_outcomes = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response", error=str(e), response=result)
            raise HTTPException(status_code=500, detail="Failed to generate expected outcomes")

        # Create the output
        return ExpectedOutcomesOutput(
            agent_name=input.agent_name,
            goals=input.goals,
            expected_outcomes=generated_outcomes["expected_outcomes"]
        )

    except Exception as e:
        logger.error("Expected outcomes generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error generating expected outcomes: {str(e)}") 