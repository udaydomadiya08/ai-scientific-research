import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def understand_goal(state: ResearchState) -> dict:
    """
    Stage 1: Goal Understanding.
    Decomposes the goal into domains and subproblems using the LLM.
    """
    goal = state.get("goal", "")
    print(f"--- STAGE 1: UNDERSTANDING GOAL ---")
    print(f"Goal: {goal}")
    
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous Scientific Discovery Engine.
Analyze the following goal: {goal}

Identify the 3 most relevant core scientific domains (e.g., Quantum Physics, Cellular Biology, Information Theory) necessary to investigate this goal.
Return ONLY a comma-separated list of the 3 domains. Do not include any other text."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"goal": goal})
    
    domains = [d.strip() for d in response.content.strip().split(",")]
    print(f"Extracted Domains: {domains}")
    
    return {"domains": domains}
