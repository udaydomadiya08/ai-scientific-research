import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def generate_manuscript(state: ResearchState) -> dict:
    """
    Stage C: Manuscript Generation.
    Writes a full academic paper in LaTeX based on the synthesis and execution results.
    """
    print(f"--- STAGE C: MANUSCRIPT GENERATION ---")
    goal = state.get("goal", "")
    hypotheses = state.get("hypotheses", [])
    experiments = state.get("experiments", [])
    execution_results = state.get("execution_results", "")
    success = state.get("execution_success", False)
    
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    feedback = state.get("reviewer_feedback", "")
    revision = state.get("revision_count", 0)
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous AI Scientist. 
Write a full, academic-grade scientific paper in LaTeX format based on the following:

Goal: {goal}
Hypotheses: {hypotheses}
Proposed Experiments: {experiments}
Experiment Execution Success: {success}
Execution Results/Output: {execution_results}

{feedback_section}

REQUIREMENTS:
1. Output MUST be valid LaTeX code starting with \documentclass{{article}}. Use packages like amsmath, graphicx, and hyperref.
2. Include sections: Abstract, Introduction, Related Work, Theoretical Framework, Methodology, Experiments, Results, Discussion, and Conclusion.
3. EXTREME DEPTH AND LENGTH: This is a full-length, professional journal submission (target: 15+ pages). DO NOT provide shallow summaries. You must write AT LEAST 800 words per section. Expand exhaustively on every concept, background theory, and methodological step.
4. MATHEMATICAL RIGOR: Include a dedicated "Theoretical Framework" section heavily populated with complex, formal mathematical equations (\begin{equation}) modeling the core hypothesis. 
5. ANTI-HALLUCINATION PROTOCOL (RESULTS): You MUST report only the exact numbers, statistics, and findings present in the `Execution Results`. Do not invent or hallucinate data that is not in the execution logs. If the experiment failed, detail the exact technical reasons and theoretical implications of the failure.
6. ANTI-HALLUCINATION PROTOCOL (CITATIONS): Do not invent fake DOIs or fake authors. If you cite a paper, it must be highly relevant and you must be confident it exists.
7. NOBEL QUALITY MANDATE: You must aim for a 10/10 peer-review score on the FIRST draft. Write with the clarity, rigor, and visionary scope of a Nobel-winning paper in Nature/Science. Ensure the narrative clearly highlights the paradigm shift.
8. Output ONLY the raw LaTeX code. Do not include any markdown wrappers (like ```latex)."""
    )
    
    feedback_section = ""
    if revision > 0 and feedback:
        feedback_section = f"PREVIOUS PEER REVIEW FEEDBACK:\n{feedback}\n\nPlease fix the manuscript to address this feedback."
    
    chain = prompt | llm
    
    context = {
        "goal": goal,
        "hypotheses": str(hypotheses),
        "experiments": str(experiments),
        "success": str(success),
        "execution_results": execution_results,
        "feedback_section": feedback_section
    }
    
    try:
        response = chain.invoke(context)
        latex = response.content
        
        # Clean up markdown if it sneaked in
        if latex.startswith("```latex"):
            latex = latex[8:]
        if latex.endswith("```"):
            latex = latex[:-3]
            
        os.makedirs("workspace", exist_ok=True)
        with open("workspace/manuscript.tex", "w") as f:
            f.write(latex.strip())
            
    except Exception as e:
        latex = f"% Failed to generate LaTeX due to: {e}"
        
    return {"manuscript_latex": latex.strip()}
