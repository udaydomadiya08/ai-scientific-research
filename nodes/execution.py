import os
import subprocess
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.state import ResearchState

def execute_experiment(state: ResearchState) -> dict:
    """
    Stage B: Experiment Execution & Auto-Debugging.
    Runs the python code in a workspace, debugs it if it crashes.
    """
    print(f"--- STAGE B: EXPERIMENT EXECUTION ---")
    code = state.get("code", "")
    
    if not code:
        return {"execution_results": "No code to execute.", "execution_success": False}
        
    os.makedirs("workspace", exist_ok=True)
    
    llm = ChatOpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url=os.environ.get("NVIDIA_BASE_URL"),
        model=os.environ.get("NVIDIA_MODEL")
    )
    
    prompt = PromptTemplate.from_template(
        """You are an Autonomous AI Scientist debugging an experiment script.
The following python script failed:
{code}

Error Log:
{error}

Fix the code. Return ONLY the fully updated, corrected Python code block starting with ```python and ending with ```."""
    )
    
    max_retries = 3
    success = False
    output_log = ""
    
    for attempt in range(max_retries):
        with open("workspace/experiment.py", "w") as f:
            f.write(code)
            
        print(f"Executing experiment.py (Attempt {attempt + 1}/{max_retries})...")
        
        try:
            result = subprocess.run(
                ["python3", "workspace/experiment.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("Execution successful!")
                output_log = result.stdout
                success = True
                break
            else:
                error_str = result.stderr
                print(f"Execution failed with error:\\n{error_str[:200]}...")
                
                # Auto-install missing packages
                module_match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_str)
                if module_match:
                    missing_module = module_match.group(1)
                    print(f"Autonomously installing missing package: {missing_module}")
                    subprocess.run(["pip", "install", missing_module], capture_output=True)
                    # Don't waste an LLM retry on a simple missing module if possible, 
                    # but we are in a loop, so we'll just let the loop continue and try the same code again.
                    # Actually, we should avoid changing the code if it's just a module error.
                    continue
                    
                # Request LLM to fix it
                chain = prompt | llm
                response = chain.invoke({"code": code, "error": result.stderr})
                text = response.content
                match = re.search(r"```python\\n(.*?)```", text, re.DOTALL)
                if match:
                    code = match.group(1)
                else:
                    code = text.replace("```python", "").replace("```", "")
        except Exception as e:
            print(f"Execution error: {e}")
            output_log = str(e)
            break
            
    if not success:
        output_log = "Execution failed after maximum retries. " + output_log
        
    return {
        "code": code, 
        "execution_results": output_log,
        "execution_success": success
    }
