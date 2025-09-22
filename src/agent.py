#GRAPH BUILDING
from nodes import (query_input, complexity_assessor, planner, agent, simple_executor, critic_evaluator, replanner, enhanced_finalizer)
from state import AgentState
from langgraph.graph import StateGraph, END
from nodes import should_continue, should_use_planning, should_replan, should_use_tools_simple_executor
from langgraph.checkpoint.memory import MemorySaver
from config import DEBUGGING_TOOL_NODE

def build_workflow(checkpointer=None) -> StateGraph[AgentState]:
    builder = StateGraph(AgentState)
    builder.add_node("INPUT", query_input)
    builder.add_node("COMPLEXITY_ASSESSOR", complexity_assessor)
    builder.add_node("PLANNING", planner)
    builder.add_node("AGENT", agent)
    builder.add_node("TOOLS", DEBUGGING_TOOL_NODE)
    builder.add_node("TOOLS_SIMPLE", DEBUGGING_TOOL_NODE)
    builder.add_node("FINALIZER", enhanced_finalizer)
    builder.add_node("SIMPLE_EXECUTOR", simple_executor)
    builder.add_node("CRITIC", critic_evaluator)
    builder.add_node("REPLANNER", replanner)


    builder.set_entry_point("INPUT")
    builder.add_edge("INPUT", "COMPLEXITY_ASSESSOR")

    builder.add_conditional_edges(
            "COMPLEXITY_ASSESSOR",
            should_use_planning,
            {"simple_executor": "SIMPLE_EXECUTOR", "planner": "PLANNING"},
        )
    builder.add_conditional_edges(
            "SIMPLE_EXECUTOR",
            should_use_tools_simple_executor,
            {"tools": "TOOLS_SIMPLE", "final_answer": "FINALIZER"},
        )
    
    builder.add_edge("TOOLS_SIMPLE", "FINALIZER")

    builder.add_edge("PLANNING", "AGENT")
    builder.add_conditional_edges(
            "AGENT",
            should_continue,
            {"tools": "TOOLS", "agent": "AGENT", "final_answer": "FINALIZER"},
        )
    builder.add_edge("TOOLS", "AGENT")
    builder.add_edge("FINALIZER", "CRITIC")
    builder.add_conditional_edges(
            "CRITIC",
            should_replan,
            {"end": END, "replan": "REPLANNER"},
        )
    builder.add_edge("REPLANNER", "AGENT")

    if checkpointer:
        system = builder.compile(checkpointer=checkpointer)
    else:
        system = builder.compile()
    return system