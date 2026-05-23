import os
import re
# pyrefly: ignore [missing-import]
import arxiv
# pyrefly: ignore [missing-import]
from semanticscholar import SemanticScholar
# pyrefly: ignore [missing-import]
from gnews import GNews
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def fetch_recent_papers() -> list[str]:
    print("Fetching recent trending papers from arXiv and Semantic Scholar...")
    queries = ["quantum biology", "cellular aging", "artificial general intelligence"]
    paper_summaries = []
    
    # 1. Fetch from arXiv
    client = arxiv.Client()
    for query in queries:
        try:
            search = arxiv.Search(
                query=f"all:{query}",
                max_results=2,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            for result in client.results(search):
                paper_summaries.append(f"[arXiv] Title: {result.title}\\nAbstract: {result.summary[:200]}...")
        except Exception as e:
            print(f"arXiv search error: {e}")
            
    return paper_summaries

def generate_scientific_goals() -> list[str]:
    print("Fetching current scientific trends from Google News...")
    google_news = GNews(max_results=5)
    news = google_news.get_news('Science')
    
    headlines = [article['title'] for article in news]
    print(f"Fetched {len(headlines)} recent science headlines.")
    
    recent_papers = fetch_recent_papers()
    print(f"Fetched {len(recent_papers)} recent research papers.")
    
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous Scientific Discovery Engine operating at a Nobel-Prize level.
Your purpose is to find the "white space" between completely disparate scientific disciplines (e.g., Quantum Mechanics + Synthetic Biology) to discover radical new paradigm shifts.

Based on current global trends and fundamental mysteries, propose exactly 3 mind-blowing, Nobel-level scientific goals.
Do NOT propose incremental improvements. Propose goals that seek to uncover hidden mechanisms, unify theories, or achieve biological/physical immortality.

Recent Science Headlines:
{headlines}

Recent Trending Research Papers (Titles & Snippets):
{recent_papers}

Analyze these sources to find what is trending and what is missing.
Propose exactly 3 goals. Format each goal on a new line starting with a number. Example:
1. Identify the epigenetic mechanism preventing entropy reversal in mammalian cellular aging.
2. Formulate a quantum computing model to simulate nitrogenase for room-temperature nitrogen fixation.
"""
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "headlines": "\n".join(headlines),
        "recent_papers": "\n\n".join(recent_papers)
    })
    
    goals = []
    for line in response.content.split("\\n"):
        line = line.strip()
        if re.match(r"^\d+\.", line):
            # Extract just the text part of the goal
            goal_text = re.sub(r"^\d+\.\s*", "", line)
            goals.append(goal_text)
            
    # Fallback if parsing fails
    if not goals:
        goals = [
            "Identify the epigenetic mechanism preventing entropy reversal in mammalian cellular aging.",
            "Formulate a quantum computing model to simulate nitrogenase for room-temperature nitrogen fixation.",
            "Discover the fundamental physical constraints on biological consciousness integration."
        ]
        
    return goals[:3]
