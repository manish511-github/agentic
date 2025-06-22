from langgraph.graph import StateGraph, END
from .models import AgentState
from .validate_goal import validate_input_node
from .tweet_fetcher import fetch_tweets_node
from .hashtag_generator import search_hashtags_node

def create_graph():
    """Create and compile the Twitter agent workflow graph"""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("search_hashtags", search_hashtags_node)
    graph.add_node("fetch_tweets", fetch_tweets_node)

    # Set entry point
    graph.set_entry_point("validate_input")
    
    # Add edges
    graph.add_edge("validate_input", "search_hashtags")
    graph.add_edge("search_hashtags", "fetch_tweets")
    graph.add_edge("fetch_tweets", END)
    
    return graph.compile()

# Create the compiled graph instance
twitter_graph = create_graph() 