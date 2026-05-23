import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def synthesize_results(state: ResearchState) -> dict:
    """
    Stage 10: Scientific Synthesis.
    Combines everything into a final research report.
    """
    print(f"--- STAGE 10: SCIENTIFIC SYNTHESIS ---")
    goal = state.get("goal", "")
    hypotheses = state.get("hypotheses", [])
    experiments = state.get("experiments", [])
    contradictions = state.get("contradictions", [])
    
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous Scientific Discovery Engine finalizing a report.
Goal: {goal}

Generated Hypotheses:
{hypotheses}

Proposed Experiments:
{experiments}

Identified Contradictions:
{contradictions}

Write a comprehensive, 3-paragraph executive scientific synthesis report based ONLY on the evidence and hypotheses above. 
Do not include any placeholders or introductory text."""
    )
    
    chain = prompt | llm
    
    context = {
        "goal": goal,
        "hypotheses": str(hypotheses),
        "experiments": str(experiments),
        "contradictions": str(contradictions)
    }
    
    try:
        response = chain.invoke(context)
        report = response.content
    except Exception as e:
        report = f"Failed to generate synthesis report due to: {e}"
        
    return {"final_synthesis": report}
