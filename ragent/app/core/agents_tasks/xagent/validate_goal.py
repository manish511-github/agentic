from .x_agent_config import logger, MIN_WAIT_TIME, MAX_WAIT_TIME, VALID_GOALS,GOAL_MAPPING
from .models import AgentState
from typing import List

async def validate_input_node(state: AgentState) -> AgentState:
    """Validate input parameters for the Twitter agent"""
    if state["goals"]:
        mapped_goals = map_agent_goals(state["goals"])
        state["goals"] = mapped_goals

    if not validate_goals(state["goals"]):
        state["error"] = f"Invalid goals. Choose from: {VALID_GOALS}"
        logger.info(f"Invalid goals. Choose from: {VALID_GOALS}")
        return state
    
    state["retries"] = 0
    state["hashtags"] = []
    state["tweets"] = []
    logger.info("Input validated", agent_name=state["agent_name"])
    return state


def map_agent_goals(agent_goals: List[str]) -> List[str]:
    """Map agent goals to Twitter agent goals"""
    mapped_goals = []
    for goal in agent_goals:
        mapped_goal = GOAL_MAPPING.get(goal.lower())
        if mapped_goal and mapped_goal not in mapped_goals:
            mapped_goals.append(mapped_goal)
    return mapped_goals if mapped_goals else ["increase brand awareness"]

def validate_goals(goals: List[str]) -> bool:
    """Validate that all goals are valid Twitter agent goals"""
    return all(goal.lower() in VALID_GOALS for goal in goals)


