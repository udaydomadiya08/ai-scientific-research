# Autonomous Scientific Discovery Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An advanced, fully autonomous AI research agent designed to perform real-world scientific discovery. Built on **LangGraph**, this system autonomously navigates the entire scientific method: generating novel hypotheses, retrieving empirical data, writing and executing Python simulations, drafting publication-ready LaTeX manuscripts, and surviving a recursive, highly rigorous automated peer-review process (The "Nobel Protocol").

## 🧠 Core Architecture

The system utilizes a multi-agent Directed Acyclic Graph (DAG) architecture where each node represents a distinct phase of the scientific process:

1. **Goal Understanding:** Parses broad, ambitious scientific goals (e.g., "Biological Immortality").
2. **Curiosity Expansion:** Formulates targeted, interdisciplinary research questions.
3. **Retrieval & RAG:** Queries **arXiv** and **Semantic Scholar**, downloading the latest empirical PDFs.
4. **Memory Construction:** Embeds papers into a local **Qdrant Vector Database** using HuggingFace Transformers.
5. **Reasoning (Einstein Node):** Applies first-principles reasoning to identify anomalies and propose radical, paradigm-shifting hypotheses.
6. **Implementation (Reality Protocol):** Writes empirical Python code to fetch and analyze real-world datasets from public APIs, strictly avoiding synthetic/fake data.
7. **Execution:** Runs the generated experimental code safely in an isolated sandbox with auto-pip install capabilities.
8. **Manuscript Generation:** Synthesizes the results into a fully formatted LaTeX scientific paper.
9. **Peer Review (Nobel Committee):** A brutal automated reviewer that fact-checks citations and demands a 10/10 paradigm shift. If rejected, it provides strict, actionable feedback and forces the system into a recursive improvement loop.

## 🚀 Features

- **The Reality Protocol:** The system is explicitly banned from using `numpy.random` for core experiments. It is mandated to fetch and process real empirical datasets via APIs.
- **Anti-Hallucination Guardrails:** Strict prompt engineering ensures the agent grounds all mathematical proofs and citations to the retrieved arXiv context and execution logs.
- **Agentic Resolve:** A self-improving recursive loop. If the peer reviewer gives a score below 10/10, the agent automatically rewrites the code and manuscript until the Nobel Committee is satisfied.
- **Nvidia NIM Integration:** Natively configured to use massive foundational models like `meta/llama-3.1-70b-instruct` via Nvidia's generative AI API, avoiding standard rate limits.

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/udaydomadiya08/ai-scientific-research.git
   cd ai-scientific-research
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API Keys:**
   Copy the example environment file and fill in your keys:
   ```bash
   cp .env.example .env
   ```

## 💻 Usage

To launch the Autonomous Scientist, simply run:
```bash
python3 main.py
```

The system will automatically fetch trending scientific headlines, propose 3 breakthrough research goals, and allow you to input a custom goal to initiate the research pipeline.

## 📂 Output

All outputs are saved to the `workspace/` directory, including:
- `experiment.py`: The generated analytical code.
- `manuscript.tex`: The publication-ready LaTeX paper.
- `peer_review.txt`: The detailed critique from the Nobel Protocol Reviewer.

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

This project was inspired by and builds upon the foundational concepts introduced by the following pioneering open-source AI research projects:
- [SakanaAI / AI-Scientist](https://github.com/sakanaai/ai-scientist)
- [HKUDS / AI-Researcher](https://github.com/hkuds/ai-researcher)
