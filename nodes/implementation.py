import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def implement_algorithm(state: ResearchState) -> dict:
    """
    Stage A: Algorithm Implementation.
    Generates Python code to run the proposed experiment.
    """
    print(f"--- STAGE A: ALGORITHM IMPLEMENTATION ---")
    experiments = state.get("experiments", [])
    
    if not experiments:
        print("No experiments found. Returning empty code.")
        return {"code": ""}
        
    experiment = experiments[0].get("proposal", "")
    feedback = state.get("reviewer_feedback", "")
    revision = state.get("revision_count", 0)
    
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous AI Scientist.
Your task is to write a complete, standalone Python script to execute the following experiment:
{experiment}

{feedback_section}

REQUIREMENTS:
1. The code must be fully self-contained and runnable.
2. It must generate some output (e.g., printing analysis results or creating a matplotlib plot saved as 'results.png').
3. REALITY PROTOCOL: You are STRICTLY FORBIDDEN from using `numpy.random` or generating synthetic fake data. You MUST use libraries like `requests`, `pandas`, `urllib`, or `biopython` to fetch REAL empirical data from public APIs (e.g., NCBI, PubChem, World Bank, arXiv, etc.).
4. NOBEL QUALITY MANDATE: Write highly advanced, optimized, and rigorous experimental code on the FIRST try so that the manuscript node receives 10/10 peer-review quality results immediately.
5. Output ONLY the raw Python code block starting with ```python and ending with ```. Do not include any explanations."""
    )
    
    feedback_section = ""
    if revision > 0 and feedback:
        feedback_section = f"PREVIOUS PEER REVIEW FEEDBACK:\n{feedback}\n\nPlease fix the algorithm to address this feedback."
    
    chain = prompt | llm
    response = chain.invoke({"experiment": experiment, "feedback_section": feedback_section})
    
    text = response.content
    
    # Extract code from markdown block
    code = ""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        code = text.replace("```python", "").replace("```", "")
        
    print("Algorithm implemented successfully.")
    return {"code": code.strip()}
