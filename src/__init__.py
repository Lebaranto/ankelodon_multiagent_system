"""ANKELODON: Core AI Agent Package.

Import key components for easy use:
from src import workflow, llm
"""

from .config import llm, TOOLS, CONFIG, TOOL_NODE, planner_llm
from .agent import workflow, build_workflow, should_continue
from .nodes import agent, planner, query_input, critique
from .schemas import AgentState, PlannerPlan, ComplexityLevel, CritiqueFeedback

__version__ = "0.1.0"
__all__ = [
    "llm", "TOOLS", "CONFIG", "TOOL_NODE", "planner_llm",
    "workflow", "build_workflow", "should_continue",
    "agent", "planner", "query_input", "critique",
    "AgentState", "PlannerPlan", "ComplexityLevel", "CritiqueFeedback",
    "__version__"
]