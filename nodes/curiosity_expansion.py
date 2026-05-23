import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def expand_curiosity(state: ResearchState) -> dict:
    """
    Stage 2: Curiosity Expansion.
    Autonomously generates deep scientific questions using the LLM.
    """
    goal = state.get("goal", "")
    domains = state.get("domains", [])
    print(f"--- STAGE 2: EXPANDING CURIOSITY ---")
    
    # Initialize the LLM using the NVIDIA endpoint
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous Scientific Discovery Engine operating with the collective curiosity of history's greatest scientific minds and researchers.
Your goal is to deeply investigate the following scientific goal: {goal}

Generate 5 profound, first-principles curiosity questions that probe the absolute fundamental limits of this domain.
Do NOT ask standard, incremental questions.
Ask "Why" until you hit the physical, biological, or mathematical constraints of reality. 
Ask questions that attempt to bridge entirely different scientific fields.

Return ONLY a numbered list of questions."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"goal": goal, "domains": ", ".join(domains) if domains else "General Science"})
    
    # Parse the questions
    questions = [q.strip("- ") for q in response.content.strip().split("\\n") if q.strip()]
    
    print(f"Generated {len(questions)} curiosity questions:")
    for q in questions:
        print(f" - {q}")
        
    return {"curiosity_questions": questions}
