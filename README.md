<!--
  ___                     _      _               _              
 / _ \                   | |    | |             | |             
/ /_\ \__ _ _ __ __ _  __| | ___| |__   ___   __| | ___  ___    
|  _  / _` | '__/ _` |/ _` |/ _ \ '_ \ / _ \ / _` |/ _ \/ __|   
| | | | (_| | | | (_| | (_| |  __/ | | | (_) | (_| |  __/\__ \   
\_| |_/\__,_|_|  \__,_|\__,_|\___|_| |_|\___/ \__,_|\___||___/   

Welcome to **Ankelodon**, a modular multi‑agent framework for complex question answering and data analysis.  
This project leverages [LangGraph](https://python.langgraph.org/) and [LangChain](https://python.langchain.com/) to orchestrate a suite of tools that can plan, execute and validate tasks on your behalf.

-->

# 🧬 Ankelodon Multi‑Agent System

**Ankelodon** is a proof‑of‑concept multi‑tool agent inspired by the GAIA evaluation framework.  
It combines planning, execution and critique to solve open‑ended queries that might involve search, file analysis, mathematics, coding or image understanding.  
By breaking down tasks into manageable steps and selecting the right tool for each job, Ankelodon aims to deliver accurate answers with verifiable evidence.

![project logo](docs/images/ankelodon_banner.png)

> *Note: The banner above is a placeholder. You can replace it with your own image placed at `docs/images/ankelodon_banner.png`.*

## 🌟 Features

### 🧠 Complexity assessment & routing

Before doing any heavy lifting, Ankelodon evaluates the incoming query to determine whether it requires planning or can be answered directly.  
Simple questions (e.g. definitions, single mathematical operations) are answered via a lightweight executor.  
Moderate and complex queries trigger the planner and agent pipeline, ensuring appropriate decomposition and tool usage【942452390578334†L22-L34】.

### 🧭 Structured planning

For non‑trivial tasks, a **planner** LLM generates a structured plan consisting of a series of steps.  
Each step has an ID, goal, selected tool, expected result and fallback strategy.  
The plan is stored as a Pydantic model (`PlannerPlan`) with strong typing for reliability【981681905155103†L82-L100】.

### 🤖 Agent execution

The **agent** node follows the plan step‑by‑step.  
For each step it first produces reasoning, then invokes the suggested tool with the appropriate inputs.  
Tool outputs are captured and fed back into subsequent reasoning.  
The agent continues until all steps are complete or an error requires replanning【981681905155103†L161-L186】.

### 🧰 Rich toolset

Ankelodon exposes a curated set of tools bound to the execution LLM:

| Tool | Purpose |
|---|---|
| `download_file_from_url` | Download files from the web by URL |
| `web_search` | Perform internet search via Tavily API |
| `arxiv_search` | Find relevant academic papers on arXiv |
| `wiki_search` | Fetch Wikipedia articles and summaries |
| `add`, `subtract`, `multiply`, `divide`, `power` | Basic arithmetic operations |
| `analyze_excel_file`, `analyze_csv_file` | Parse spreadsheets and compute statistics |
| `analyze_docx_file`, `analyze_pdf_file`, `analyze_txt_file` | Extract and summarise document content |
| `vision_qa_gemma` | Answer questions about images using a vision model |
| `safe_code_run` | Execute Python code securely in an isolated environment |

These tools are loaded into a `ToolNode` and passed to the agent for use during execution【774776463100239†L10-L14】.

### 📝 Comprehensive reporting & critique

After the agent finishes, a deterministic LLM generates a structured execution report.  
This report summarises the query, steps taken, key findings, sources used, and the final answer.  
A separate **critic** LLM evaluates the report for completeness, accuracy, methodology and evidence, scoring it out of 10 and suggesting improvements if necessary【981681905155103†L459-L525】.  
The system may then replan and re‑execute until the answer meets quality thresholds.

## 🏗 Architecture

Ankelodon is built as a directed acyclic graph of nodes. The high‑level flow is:

1. **INPUT** – Receive the user query and optional files.  
2. **COMPLEXITY_ASSESSOR** – Classify the query as simple, moderate or complex and decide whether to plan.  
3. **PLANNING** – Generate a multi‑step plan when needed, using examples and strict rules about tool usage and numerical computation.  
4. **AGENT** – Iterate through the plan: reason about each step, call a tool, capture results and update state.  
5. **TOOLS** – Execute selected tools via a unified `ToolNode`.  
6. **FINALIZER** – Consolidate the execution into a report and extract a formatted final answer.  
7. **CRITIC** – Score the report and decide whether to accept or trigger the **REPLANNER**.  

The graph is compiled using LangGraph’s `StateGraph` API and is flexible enough to be extended with new nodes or tools【942452390578334†L8-L50】.

## 🚀 Getting started

### Prerequisites

This project targets **Python 3.10+**. You’ll need API keys or credentials for any external services (e.g. OpenAI, Tavily, Gemini) used by tools.  
Assuming you have a virtual environment activated:

```bash
pip install langchain==0.1.* langgraph openai google-generativeai
# plus any other packages referenced in tools (pandas, numpy, pillow, tldextract, etc.)
```

### Running a simple query

The entry point is the `build_workflow` function in `src/agent.py`. It returns a compiled system you can invoke with a dictionary representing the agent state.  
A minimal example:

```python
from src.agent import build_workflow

# Initialize the graph
system = build_workflow()

# Build the initial state
state = {
    "query": "What is the square root of 144?",
    "messages": [],
    "files": [],
    "iteration_count": 0,
    "max_iterations": 3
}

# Invoke the system and get the result
result = system.invoke(state)
print(result.get("final_answer"))  # should output: FINAL ANSWER: 12
```

For more complex tasks involving file uploads or web searches, provide file paths in the `files` list and ensure appropriate API keys are set in the environment.

### Notebooks & examples

There are example notebooks under `src/` and `test_folder/` demonstrating how to test the agent with sample queries and data.  
Feel free to explore and adapt them to your own scenarios.

## 🛣 Roadmap & GAIA adaptation

- Integrate unit conversion, date arithmetic and table operations to handle GAIA evaluation tasks out‑of‑the‑box.  
- Add question‑clarification and error‑recovery loops to minimise unnecessary replanning.  
- Streamline the tool list by removing unused tools and grouping related operations.  
- Improve caching of external calls (e.g. web search, downloads) to speed up repeated queries.  
- Expand the test suite and add continuous integration.

## 🤝 Contributing

Contributions are welcome! If you find a bug or have an idea for improvement, feel free to open an issue or a pull request.  
When adding new tools or nodes, please ensure they adhere to the structured planning and execution patterns shown here, and update the tests accordingly.

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

*Ankelodon is a work in progress. Your feedback and use‑cases will help shape its future. Happy hacking!* 🦾
