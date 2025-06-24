from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.validate_goals import validate_input_node
from .nodes.search_posts_directly import search_posts_directly_node
from .nodes.search_subreddits import search_subreddits_node
from .nodes.fetch_posts import fetch_posts_node, fetch_basic_post_nodes

def create_reddit_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("search_posts_directly_node", search_posts_directly_node)
    graph.add_node("search_subreddits_node", search_subreddits_node)
    graph.add_node("fetch_posts_node", fetch_posts_node)
    graph.set_entry_point("validate_input")
    graph.add_edge("validate_input", "search_posts_directly_node")
    graph.add_edge("search_posts_directly_node", "search_subreddits_node")
    graph.add_edge("search_subreddits_node", "fetch_posts_node")
    graph.add_edge("fetch_posts_node", END)
    return graph.compile()

def create_basic_reddit_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("fetch_basic_post_node", fetch_basic_post_nodes)
    graph.set_entry_point("fetch_basic_post_node")
    graph.add_edge("fetch_basic_post_node", END)
    return graph.compile()

reddit_graph = create_reddit_graph()
basic_redit_agent = create_basic_reddit_graph() 