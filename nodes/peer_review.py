import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def automated_peer_review(state: ResearchState) -> dict:
    """
    Stage D: Automated Peer Review.
    Simulates a rigorous peer review process.
    """
    print(f"--- STAGE D: AUTOMATED PEER REVIEW ---")
    latex = state.get("manuscript_latex", "")
    
    if not latex:
        return {"peer_review": "No manuscript generated to review."}
        
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are the Nobel Prize Committee and the harshest peer reviewer in the world (e.g., Nature, Science).
Review the following academic manuscript:
{latex}

Provide a detailed peer review based on the "NOBEL PROTOCOL":
- If this research is "incremental", "safe", or "standard", you MUST reject it and give it a low score (< 7).
- To achieve a 10/10, the manuscript MUST propose a verified paradigm shift, a fundamental breakthrough, or a radical unification of fields that pushes humanity forward.

Include:
1. Summary of Contributions
2. Paradigm-Shift Analysis (Is this revolutionary or incremental?)
3. Strengths & Weaknesses
4. Final Recommendation (Accept / Reject / Major Revision)
5. Overall Score (out of 10)

Be highly critical, analytical, and fair. 
IMPORTANT: Your response MUST end with exactly this format:
SCORE: X/10
(where X is an integer from 1 to 10)."""
    )
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"latex": latex})
        review = response.content
    except Exception as e:
        review = f"Review failed: {e}\nSCORE: 0/10"
        
    with open("workspace/peer_review.txt", "w") as f:
        f.write(review)
        
    # Parse the score
    import re
    score_match = re.search(r"SCORE:\s*(\d+)/10", review)
    score = int(score_match.group(1)) if score_match else 0
        
    return {"peer_review": review, "peer_review_score": score, "reviewer_feedback": review}
