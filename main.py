import argparse
import shutil
import os
from dotenv import load_dotenv
from core.graph import build_graph
from core.goal_generator import generate_scientific_goals

def cleanup_temp_files():
    """Wipe all temporary files after research is delivered."""
    dirs_to_clean = ["data", "qdrant_data"]
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Cleaned: {d}/")
    # Clean generated PNGs from root
    for f in os.listdir("."):
        if f.endswith(".png"):
            os.remove(f)
            print(f"  Cleaned: {f}")
    print("[CLEANUP COMPLETE] System is fresh for next run.\n")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Autonomous Scientific Discovery Intelligence System")
    parser.add_argument("--goal", type=str, required=False, help="The scientific goal, problem, or mystery to investigate")
    args = parser.parse_args()

    print("==================================================")
    print("= AUTONOMOUS SCIENTIFIC DISCOVERY INTELLIGENCE   =")
    print("==================================================")
    
    goal = args.goal
    
    if not goal:
        print("\n[AUTONOMOUS GOAL GENERATION PHASE]")
        goals = generate_scientific_goals()
        print("\nI have analyzed current scientific trends and formulated the following breakthrough goals:")
        for i, g in enumerate(goals, 1):
            print(f"{i}. {g}")
        print("4. [Enter your own custom goal]")
        
        while True:
            try:
                choice = int(input("\nPlease select a goal (1-4) to begin research: "))
                if 1 <= choice <= 3:
                    goal = goals[choice - 1]
                    break
                elif choice == 4:
                    goal = input("Enter your custom scientific goal: ").strip()
                    if goal:
                        break
            except ValueError:
                pass
            print("Invalid choice. Please enter a number between 1 and 4.")

    print(f"\n[INITIATING PIPELINE] Goal: {goal}\n")
    
    app = build_graph()
    
    # Initialize the state
    inputs = {"goal": goal}
    
    # Run the graph and collect final state
    final_state = app.invoke(inputs)
    
    print("\n==================================================")
    print("= FINAL RESULTS                                  =")
    print("==================================================")
    print("MANUSCRIPT (LaTeX):")
    print(final_state.get("manuscript_latex", "No manuscript generated.")[:500] + "...\n(truncated)\n")
    print("PEER REVIEW:")
    print(final_state.get("peer_review", "No peer review generated."))
    
    # Cleanup temp files, keep workspace/ deliverables
    print("\n[CLEANUP] Removing temporary files...")
    cleanup_temp_files()

if __name__ == "__main__":
    main()

