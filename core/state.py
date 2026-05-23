from typing import TypedDict, List, Dict, Any

class ResearchState(TypedDict):
    goal: str
    domains: List[str]
    curiosity_questions: List[str]
    retrieved_papers: List[Dict[str, Any]]
    mechanisms: List[str]
    contradictions: List[str]
    hypotheses: List[Dict[str, Any]]
    experiments: List[Dict[str, Any]]
    cross_domain_insights: List[str]
    final_synthesis: str
    error: str

    # AI-Scientist / AI-Researcher Advanced Fields
    code: str
    execution_results: str
    execution_success: bool
    manuscript_latex: str
    peer_review: str
    peer_review_score: int
    revision_count: int
    reviewer_feedback: str
