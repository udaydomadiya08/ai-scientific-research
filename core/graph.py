from langgraph.graph import StateGraph, END
from core.state import ResearchState
from nodes.goal_understanding import understand_goal
from nodes.curiosity_expansion import expand_curiosity
from nodes.retrieval import retrieve_papers
from nodes.parsing import parse_and_chunk
from nodes.memory import construct_memory
from nodes.reasoning import cross_paper_reasoning
from nodes.synthesis import synthesize_results
from nodes.implementation import implement_algorithm
from nodes.execution import execute_experiment
from nodes.manuscript import generate_manuscript
from nodes.peer_review import automated_peer_review

def build_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)
    
    # Add Nodes
    workflow.add_node("understand_goal", understand_goal)
    workflow.add_node("expand_curiosity", expand_curiosity)
    workflow.add_node("retrieve_papers", retrieve_papers)
    workflow.add_node("parse_and_chunk", parse_and_chunk)
    workflow.add_node("construct_memory", construct_memory)
    workflow.add_node("cross_paper_reasoning", cross_paper_reasoning)
    workflow.add_node("synthesize_results", synthesize_results)
    
    workflow.add_node("implement_algorithm", implement_algorithm)
    workflow.add_node("execute_experiment", execute_experiment)
    workflow.add_node("generate_manuscript", generate_manuscript)
    workflow.add_node("automated_peer_review", automated_peer_review)
    
    # Define Routing Logic
    def review_router(state: ResearchState) -> str:
        score = state.get("peer_review_score", 0)
        revisions = state.get("revision_count", 0)
        
        if score >= 10 or revisions >= 3:
            return END
        else:
            print(f"\\n[AGENTIC RESOLVE] Paper scored {score}/10. Initiating revision {revisions + 1}/3...")
            return "implement_algorithm"
    
    # Define Edges
    workflow.set_entry_point("understand_goal")
    
    workflow.add_edge("understand_goal", "expand_curiosity")
    workflow.add_edge("expand_curiosity", "retrieve_papers")
    workflow.add_edge("retrieve_papers", "parse_and_chunk")
    workflow.add_edge("parse_and_chunk", "construct_memory")
    workflow.add_edge("construct_memory", "cross_paper_reasoning")
    workflow.add_edge("cross_paper_reasoning", "synthesize_results")
    
    workflow.add_edge("synthesize_results", "implement_algorithm")
    workflow.add_edge("implement_algorithm", "execute_experiment")
    workflow.add_edge("execute_experiment", "generate_manuscript")
    workflow.add_edge("generate_manuscript", "automated_peer_review")
    
    # Add Conditional Edge for Loop
    workflow.add_conditional_edges("automated_peer_review", review_router)
    
    return workflow.compile()
