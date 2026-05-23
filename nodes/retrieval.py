import arxiv
from core.state import ResearchState

def retrieve_papers(state: ResearchState) -> dict:
    """
    Stage 3: Research Paper Retrieval.
    Dynamically searches arXiv based on goal complexity and curiosity questions.
    Retrieves more papers for broader/harder goals.
    """
    goal = state.get("goal", "")
    questions = state.get("curiosity_questions", [])
    domains = state.get("domains", [])
    print(f"--- STAGE 3: RETRIEVING PAPERS ---")
    
    # Build multiple search queries from goal, domains, and curiosity questions
    queries = [goal]
    for d in domains[:3]:
        queries.append(f"{d} {goal}")
    if isinstance(questions, list):
        for q in questions[:2]:
            # Take first meaningful phrase from each question
            short_q = " ".join(q.split()[:6])
            queries.append(short_q)
    
    # Deduplicate
    queries = list(dict.fromkeys(queries))
    
    # Dynamic paper count: 10 per query, up to 50 total
    papers_per_query = 10
    max_total = 50
    
    client = arxiv.Client()
    papers = []
    seen_titles = set()
    
    for q in queries:
        if len(papers) >= max_total:
            break
        try:
            search = arxiv.Search(
                query=q,
                max_results=papers_per_query,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in client.results(search):
                if result.title not in seen_titles and len(papers) < max_total:
                    seen_titles.add(result.title)
                    papers.append({
                        "title": result.title,
                        "abstract": result.summary,
                        "authors": [a.name for a in result.authors],
                        "url": result.pdf_url
                    })
        except Exception as e:
            print(f"Query '{q}' failed: {e}")
            continue
        
    print(f"Retrieved {len(papers)} papers from {len(queries)} queries.")
    return {"retrieved_papers": papers}
