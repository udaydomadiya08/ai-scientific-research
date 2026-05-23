import os
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def cross_paper_reasoning(state: ResearchState) -> dict:
    """
    Stage 7, 8, 9: Cross-Paper Reasoning, Hypothesis Generation, Experimental Design.
    Retrieves facts from Qdrant and prompts the LLM to generate hypotheses.
    """
    goal = state.get("goal", "")
    questions = state.get("curiosity_questions", [])
    
    print(f"--- STAGE 7-9: REASONING & HYPOTHESIS GENERATION ---")
    
    try:
        client = QdrantClient(path="qdrant_data")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        collection_name = "scientific_memory"
        
        # We will use the first curiosity question as a query
        query = questions[0] if questions else goal
        query_vector = embeddings.embed_query(query)
        
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5
        )
        context = "\n".join([hit.payload["text"] for hit in search_result])
    except Exception as e:
        print(f"Failed to query Qdrant: {e}")
        context = "No relevant context found."
        
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous Scientific Discovery Engine operating with the collective genius of history's greatest scientific minds and researchers.
Your goal is to investigate: {goal}

We extracted the following context from recent research papers:
{context}

Based on this evidence, you must find the ANOMALIES. Look for paradoxes, contradictions, or missing links in the current literature.
Use FIRST-PRINCIPLES reasoning to generate 1 radical, paradigm-shifting hypothesis that challenges the status quo. Do NOT propose incremental science.
Then, propose 1 specific experimental design to prove this radical hypothesis.

Format your output as exactly 3 sections:
CONTRADICTION: [Describe the paradox or gap in the current literature]
HYPOTHESIS: [Your radical, paradigm-shifting theory]
EXPERIMENT: [Your method to prove it]"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"goal": goal, "context": context})
    
    text = response.content
    contradictions = []
    hypotheses = []
    experiments = []
    
    # Parse the response safely
    current_section = None
    buffer = []
    
    for line in text.split("\n"):
        if "CONTRADICTION:" in line:
            current_section = "C"
            buffer.append(line.replace("CONTRADICTION:", "").strip())
        elif "HYPOTHESIS:" in line:
            if current_section == "C": contradictions.append(" ".join(buffer).strip())
            current_section = "H"
            buffer = [line.replace("HYPOTHESIS:", "").strip()]
        elif "EXPERIMENT:" in line:
            if current_section == "H": hypotheses.append({"hypothesis": " ".join(buffer).strip()})
            current_section = "E"
            buffer = [line.replace("EXPERIMENT:", "").strip()]
        elif current_section:
            buffer.append(line.strip())
            
    if current_section == "E":
        experiments.append({"proposal": " ".join(buffer).strip()})
        
    return {
        "contradictions": contradictions,
        "hypotheses": hypotheses,
        "experiments": experiments
    }
