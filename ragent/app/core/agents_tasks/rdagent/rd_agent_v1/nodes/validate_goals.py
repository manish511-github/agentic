from ..state import RedditAgentState
import structlog

logger = structlog.get_logger()

def map_agent_goals(agent_goals):
    GOAL_MAPPING = {
        "lead_generation": "grow web traffic",
        "brand_awareness": "increase brand awareness",
        "engagement": "engage potential customers",
        "support": "engage potential customers"
    }
    mapped_goals = []
    for goal in agent_goals:
        mapped_goal = GOAL_MAPPING.get(goal.lower())
        if mapped_goal and mapped_goal not in mapped_goals:
            mapped_goals.append(mapped_goal)
    return mapped_goals if mapped_goals else ["increase brand awareness"]

async def validate_input_node(state: RedditAgentState) -> RedditAgentState:
    valid_goals = ["increase brand awareness", "engage potential customers", "grow web traffic"]
    if state["goals"]:
        mapped_goals = map_agent_goals(state["goals"])
        state["goals"] = mapped_goals

    if not all(goal.lower() in valid_goals for goal in state["goals"]):
        state["error"] = f"Invalid goals. Choose from: {valid_goals}"
        logger.info(f"Invalid goals. Choose from: {valid_goals}")
        return state
    
    state["retries"] = 0
    state["subreddits"] = []
    state["posts"] = []
    logger.info("Input validated", agent_name=state["agent_name"])
    return state 