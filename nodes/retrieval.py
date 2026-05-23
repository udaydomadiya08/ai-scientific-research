import arxiv
from core.state import ResearchState

def retrieve_papers(state: ResearchState) -> dict:
    """
    Stage 3: Research Paper Retrieval.
    Searches arXiv based on the domains and goal.
    """
    goal = state.get("goal", "")
    print(f"--- STAGE 3: RETRIEVING PAPERS ---")
    
    # We will search for a generic term based on the goal
    query = goal.split()[0] if goal else "physics"
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "url": result.pdf_url
        })
        
    print(f"Retrieved {len(papers)} papers.")
    return {"retrieved_papers": papers}
