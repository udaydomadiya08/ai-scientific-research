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
        
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=5
        )
        context = "\n".join([hit.payload["text"] for hit in search_result.points])
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

CRITICAL: You are aiming for a 10/10 NOBEL-PRIZE quality theory on your FIRST attempt. Do not hold back. Be visionary but scientifically grounded.

ULTIMATE GOAL: Do not generate detail for the sake of detail. Provide immense, ground-breaking scientific value. Your discoveries MUST push humanity forward. Be extremely aggressive in challenging the boundaries of current knowledge to achieve true breakthroughs.

Format your output as a SINGLE VALID JSON OBJECT with exactly 3 keys: "contradiction", "hypothesis", and "experiment". Do not output any markdown formatting like ```json. Just raw JSON."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"goal": goal, "context": context})
    
    text = response.content
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
        
    import json
    try:
        data = json.loads(text.strip())
        contradictions = [data.get("contradiction", "")]
        hypotheses = [{"hypothesis": data.get("hypothesis", "")}]
        experiments = [{"proposal": data.get("experiment", "")}]
    except json.JSONDecodeError:
        print("Failed to parse JSON. Falling back to raw text.")
        contradictions = ["Failed to parse contradiction."]
        hypotheses = [{"hypothesis": text}]
        experiments = [{"proposal": "Design an experiment to test the hypothesis above."}]
        
    return {
        "contradictions": contradictions,
        "hypotheses": hypotheses,
        "experiments": experiments
    }
