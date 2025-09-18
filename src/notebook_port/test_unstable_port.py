# === AUTO-GENERATED FROM test_.ipynb (do not edit logic) ===
# Only additive imports below to resolve package paths.
import sys, os
from pathlib import Path as _Path

# Ensure project root is importable when running as a module
_CUR = _Path(__file__).resolve()
_SRC = _CUR.parent.parent
_ROOT = _SRC.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Prefer package-qualified imports; leave original notebook imports untouched below.
try:
    from src.prompts import *          # noqa: F401,F403
    from src.schemas import *          # noqa: F401,F403
    from src.tools import *            # noqa: F401,F403
    from src.tools.code_interpreter import safe_code_run  # noqa: F401
except Exception:
    # Fallbacks if executed inside src as working directory
    pass


# === CELL 0 FROM NOTEBOOK ===
import math
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import uuid
from prompts import *
from schemas import *
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

import os, io, json, base64
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

# pip install google-generativeai pillow
import google.generativeai as genai
from PIL import Image
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
import pandas as pd
from IPython.display import display, Image
from langchain_community.document_loaders import DataFrameLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
import pickle 


from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from typing import List, TypedDict, Annotated, Literal, Optional, Union

from langgraph.graph import StateGraph, END

load_dotenv()
import os
import json
import re
import operator

from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore() #сохраняем состояние между запусками

from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from PIL import Image, ImageStat, ExifTags
import pandas as pd


#TOOLS

from tools import (web_search, arxiv_search, wiki_search, add, subtract, multiply, divide, power, 
analyze_csv_file, analyze_docx_file, analyze_pdf_file, analyze_txt_file, analyze_image_file, vision_qa_gemma, analyze_excel_file, preprocess_files, save_and_read_file, download_file_from_url)

from code_interpreter import safe_code_run


# === CELL 1 FROM NOTEBOOK ===

def clean_message_history(messages):
    """
    Очищает историю сообщений от неполных циклов tool_calls/responses.
    Удаляет AIMessage с tool_calls, если нет соответствующих ToolMessage.
    """
    cleaned_messages = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        # Если это AIMessage с tool_calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Ищем соответствующие ToolMessage
            tool_call_ids = {tc['id'] for tc in msg.tool_calls}
            found_responses = set()
            
            # Проверяем следующие сообщения на наличие ответов
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                if messages[j].tool_call_id in tool_call_ids:
                    found_responses.add(messages[j].tool_call_id)
                j += 1
            
            # Если все tool_calls имеют ответы, добавляем весь блок
            if found_responses == tool_call_ids:
                # Добавляем AIMessage и все соответствующие ToolMessage
                cleaned_messages.append(msg)
                for k in range(i + 1, j):
                    cleaned_messages.append(messages[k])
                i = j
            else:
                # Пропускаем неполный блок
                print(f"Removing incomplete tool call block: {tool_call_ids - found_responses}")
                i = j
        else:
            # Обычное сообщение - добавляем
            cleaned_messages.append(msg)
            i += 1
    
    return cleaned_messages

# === CELL 2 FROM NOTEBOOK ===
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.25)
TOOLS = [download_file_from_url, web_search, arxiv_search, wiki_search, add, subtract, multiply, divide, power, analyze_excel_file, analyze_csv_file, analyze_docx_file, analyze_pdf_file, analyze_txt_file, analyze_image_file, vision_qa_gemma, safe_code_run]

# === CELL 3 FROM NOTEBOOK ===
llm_with_tools = llm.bind_tools(TOOLS)
config = {"configurable": {"thread_id": "1"}, "recursion_limit" : 50}
TOOL_NODE = ToolNode(TOOLS)
planner_llm = llm.with_structured_output(PlannerPlan)

class AgentState(MessagesState):
    query: str
    final_answer: str
    plan: Optional[PlannerPlan]
    complexity_assessment: ComplexityLevel
    current_step: int
    reasoning_done: bool
    messages : Annotated[Sequence[BaseMessage], add_messages]
    files: List[str]
    file_contents: Dict[str, Any]
    critique_feedback: Optional[CritiqueFeedback]
    iteration_count :int
    max_iterations: int
    execution_report : ExecutionReport



def query_input(state : AgentState) -> AgentState:
    print("=== USER QUERY TRANSFERED TO AGENT ===")

    files = state.get("files", [])
    if files:
        print(f"Processing {len(files)} files:")
        file_info = preprocess_files(files)
    
        for file_path, info in file_info.items():
            print(f"  - {file_path}: {info['type']} ({info['size']} bytes) -> {info['suggested_tool']}")

        state["file_contents"] = file_info
        file_context = "\n\n=== AVAILABLE FILES FOR ANALYSIS ===\n"
        for file_path, info in file_info.items():
            filename = os.path.basename(file_path)
            file_context += f"File: {filename}\n"
            file_context += f"  - Type: {info['type']}\n"  
            file_context += f"  - Size: {info['size']} bytes\n"
            file_context += f"  - Suggested tool: {info['suggested_tool']}\n"
            if info.get("preview"):
                file_context += f"  - Preview: {info['preview']}\n"
            file_context += "\n"
        
        # Добавляем инструкции по работе с файлами
        file_context += "IMPORTANT: Use the suggested tools to analyze these files before processing their data.\n"
        file_context += "File paths are available in the agent state and can be passed directly to analysis tools.\n"
        
        original_query = state.get("query", "")
        state["query"] = original_query + file_context
    return state


def planner(state : AgentState) -> AgentState:
    sys_stack = [
            SystemMessage(content=SYSTEM_PROMPT_PLANNER.strip()),
            HumanMessage(content=state["query"]),
        ]
    plan: PlannerPlan = planner_llm.invoke(sys_stack)
    
    print("=== GENERATED PLAN ===")
    return {"messages" : sys_stack + state["messages"],
            "plan": plan,
            "current_step ": 0,
            "reasoning_done": False}

def agent(state: AgentState) -> AgentState:
    
    """
    sys_msg = SystemMessage(
        content=SYSTEM_EXECUTOR_PROMPT.strip().format(
            plan=json.dumps(state["plan"], indent=2)
        )
    )
    """
    current_step = state.get("current_step", 0)
    reasoning_done = state.get("reasoning_done", False)
    plan = state.get("plan", {})
    steps = state["plan"].steps

    print(f"=== AGENT DEBUG ===")
    print(f"Current step: {current_step}")
    print(f"Reasoning done: {reasoning_done}")
    print(f"Plan exists: {plan is not None}")
    print(f"Total steps in plan: {len(plan.steps) if plan else 'No plan'}")

    if not plan or not hasattr(plan, 'steps') or not plan.steps:
        print("ERROR: No valid plan found!")
        return {
            "messages": state["messages"] + [AIMessage(content="No valid plan available. <FINAL_ANSWER>")],
            "reasoning_done": False
        }
    
    steps = plan.steps
    
    if current_step >= len(steps):
        print("All plan steps completed, moving to finalization")
        return {
            "messages": state["messages"] + [AIMessage(content="All steps completed. <FINAL_ANSWER>")],
            "reasoning_done": False
        }

    current_step_info = steps[current_step]
    print(f"Executing step {current_step + 1}: {current_step_info.description}")

    if not reasoning_done:

        # ✅ ДОБАВЛЕНО: Специальный контекст для файлов
        file_context = ""
        file_contents = state.get("file_contents", {})
        if file_contents:
            file_context = "\n\nAVAILABLE FILES IN CURRENT SESSION:\n"
            for filepath, info in file_contents.items():
                filename = os.path.basename(filepath)
                file_context += f"- {filename}: {info['type']} file, suggested tool: {info['suggested_tool']}\n"
                file_context += f"  Path: {filepath}\n"

        reasoning_prompt = f"""
        {SYSTEM_EXECUTOR_PROMPT}
        
        CURRENT TASK: You must perform reasoning for step {current_step + 1}.
        
        STEP INFO: {current_step_info}\n\n

        FILE CONTEXT: {file_contents}
        
        CRITICAL: You MUST output your reasoning in <REASONING> tags, but DO NOT call any tools yet.
        Explain what you need to do and why, then end your response.

        REASONING IS IMPERATIVE BEFORE ANY TOOL CALLS.
        """

        sys_msg = SystemMessage(content = reasoning_prompt)
        stack = [sys_msg] + state["messages"]

        step = llm.invoke(stack)
        print("=== REASONING STEP ===")
        print(step.content)

        return {
            "messages" : state["messages"] + [step],
            "reasoning_done" : True
        }
    
    else:
        tool_prompt = f"""
        Now execute the tool for step {current_step + 1}.
        
        You have already done the reasoning. Now call the appropriate tool with the correct parameters.
        Available file paths: {list(state.get("file_contents", {}).keys())}\n
        IMPORTANT NOTE: IF YOU DECIDED TO USE safe_code_run, MAKE SURE TO FINISH CALCULATIONS WITH print() or saving to a variable NAMED 'result' so that the output can be captured!
        AVAILABLE TOOLS: {', '.join([tool.name for tool in TOOLS])}
        """ 

        sys_msg = SystemMessage(content=tool_prompt)
        stack = [sys_msg] + state["messages"]  # Берем последние сообщения включая reasoning
        
        # Используем модель С инструментами для выполнения
        step = llm_with_tools.invoke(stack)
        print("=== TOOL EXECUTION ===")
        print(f"Tool calls: {step.tool_calls}")
        
        return {
            "messages": state["messages"] + [step],
            "current_step": current_step + 1 if step.tool_calls else current_step,
            "reasoning_done": False  # Сбрасываем для следующего шага
        }


def should_continue(state : AgentState) -> bool:
    
    last_message = state["messages"][-1]
    reasoning_done = state.get("reasoning_done", False)
    plan = state.get("plan", None)
    current_step = state.get("current_step", 0)

    if plan and current_step >= len(plan.steps):
        return "final_answer"


    if hasattr(last_message, "content") and "<FINAL_ANSWER>" in last_message.content:
        return "final_answer"
    elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools" 
    elif not reasoning_done and hasattr(last_message, 'content') and "<REASONING>" in last_message.content:
        # Reasoning выполнен, но инструменты еще не вызваны
        return "agent"
    elif reasoning_done:
        # Reasoning выполнен, теперь нужно вызвать инструменты
        return "agent"
    else:
        # Нужно сделать reasoning
        return "agent"

# 6. Добавить отладочную информацию в TOOL_NODE
class DebuggingToolNode(ToolNode):
    def __init__(self, tools):
        super().__init__(tools)
    
    def __call__(self, state):
        print("=== TOOL EXECUTION STARTED ===")
        result = super().__call__(state)
        print("=== TOOL EXECUTION COMPLETED ===")
        return result

DEBUGGING_TOOL_NODE = DebuggingToolNode(TOOLS)



"""
def summary(state : AgentState) -> AgentState:
    print("=== FINAL ANSWER ===")
    summarizer_prompt = 
    Now you have to provide final answer for the user query : {query}
    In messages below you have all the context you need.

    YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, Apply the rules above for each element (number or string), ensure there is exactly one space after each comma.
    Your answer should only start with "FINAL ANSWER: ", then follows with the answer.

    Here is the context:
    {messages}

    REMEMBER AND STRICTLY FOLLOW THE FORMATTING RULES ABOVE. ALWAYS USE THIS FORMAT:
    FINAL ANSWER: ...
    

    state["final_answer"] = llm.invoke([SystemMessage(content=summarizer_prompt.strip().format(query=state["query"], messages = state["messages"]))])
    return state
"""

def enhanced_finalizer(state: AgentState) -> AgentState:
    """Generate comprehensive execution report for critic evaluation."""
    print("=== GENERATING EXECUTION REPORT ===")
    
    # Extract tool execution information
    tools_executed = []
    data_sources = []
    
    for msg in state["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tools_executed.append(ToolExecution(
                tool_name=tool_call['name'],
                arguments=str(tool_call['args']),
                call_id=tool_call['id']
            ))
        
        # Extract data sources from tool results
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            # Look for URLs, file names, or other sources
            import re
            urls = re.findall(r'https?://[^\s]+', msg.content)
            data_sources.extend(urls)
    
    # Get plan information if available
    plan = state.get("plan")
    approach_used = "Direct execution"
    assumptions_made = []
    
    if plan:
        approach_used = f"{plan.task_type} approach with {len(plan.steps)} steps"
        assumptions_made = plan.assumptions
    
    # Generate structured report (КОСТЫЛЬ ЗДЕСЬ!)
    report_generator_prompt = f"""
    Generate a comprehensive execution report for the following query processing:

    ORIGINAL QUERY: {state['query']}
    
    EXECUTION CONTEXT:
    - Complexity Level: {state.get('complexity_assessment', {}).level}
    - Plan Used: {plan if plan else {}}
    - Tools Executed: {tools_executed}
    - Available Files: {list(state.get('file_contents', {}).keys())}
    
    CONVERSATION HISTORY:
    {[msg.content[:200] + "..." if len(msg.content) > 200 else msg.content 
      for msg in state['messages'][-5:]]}  # Last 5 messages for context
    
    Based on this information, create a structured execution report that includes:
    1. Query summary
    2. Approach used
    3. Key findings from the execution
    4. Data sources used
    5. Your confidence level in the results
    6. Any limitations or caveats
    7. The final answer
    
    Be thorough but concise. This report will be evaluated by a critic for quality assurance.
    """
    
    report_llm = llm.with_structured_output(ExecutionReport)
    
    execution_report = report_llm.invoke([
        SystemMessage(content=report_generator_prompt),
        HumanMessage(content="Generate the execution report.")
    ])
    
    print(f"Report generated - Confidence: {execution_report.confidence_level}")
    print(f"Key findings: {len(execution_report.key_findings)}")
    print(f"Data sources: {len(execution_report.data_sources)}")
    
    # Format final answer for user
    formatted_answer = format_final_answer(execution_report, state.get('complexity_assessment', {}))
    print(execution_report)
    return {
        "execution_report": execution_report,
        "final_answer": formatted_answer
    }

def format_final_answer(report: ExecutionReport, complexity: dict) -> str:
    """Format the final answer based on complexity and report content."""
    
    if complexity.level == 'simple':
        # For simple queries, just return the answer
        return f"FINAL ANSWER: {report.final_answer}"
    
    # For complex queries, provide more detailed response
    formatted = f"""FINAL ANSWER: {report.final_answer}

SUMMARY:
{report.query_summary}

KEY FINDINGS:
{chr(10).join(f"• {finding}" for finding in report.key_findings)}"""
    
    if report.data_sources:
        formatted += f"""

SOURCES:
{chr(10).join(f"• {source}" for source in report.data_sources[:5])}"""  # Limit to 5 sources
    
    if report.limitations:
        formatted += f"""

LIMITATIONS:
{chr(10).join(f"• {limitation}" for limitation in report.limitations)}"""
    
    return formatted


def complexity_assessor(state: AgentState) -> AgentState:
    """Assess query complexity and determine if planning is needed."""
    print("=== COMPLEXITY ASSESSMENT ===")
    
    complexity_llm = llm.with_structured_output(ComplexityLevel)
    
    assessment_message = [
        SystemMessage(content=COMPLEXITY_ASSESSOR_PROMPT.strip()),
        HumanMessage(content=f"Query: {state['query']}")
    ]
    
    assessment = complexity_llm.invoke(assessment_message)
    
    print(f"Complexity: {assessment.level}")
    print(f"Needs planning: {assessment.needs_planning}")
    print(f"Reasoning: {assessment.reasoning}")
    
    return {
        "complexity_assessment": assessment,
        "messages": state["messages"] + assessment_message
    }


def simple_executor(state: AgentState) -> AgentState:
    """Handle simple queries directly without planning."""
    print("=== SIMPLE EXECUTION ===")
    
    # For simple queries, use the LLM with tools directly
    simple_prompt = f"""
    Answer this simple query directly and efficiently: {state['query']}
    
    You have access to tools if needed, but try to answer directly when possible.
    If you need files, they are available at: {list(state.get('file_contents', {}).keys())}
    
    Provide a clear, concise answer.
    """
    
    response = llm_with_tools.invoke([
        SystemMessage(content=simple_prompt),
        HumanMessage(content=state['query'])
    ])
    
    return {
        "messages": state["messages"] + [response],
        "final_answer": response.content
    }


def should_use_planning(state: AgentState) -> str:
    """Route based on complexity assessment."""
    complexity = state["complexity_assessment"]
    
    if complexity.level == "simple" and not complexity.needs_planning:
        return "simple_executor"
    else:
        return "planner"
    
"""    
def critic_evaluator(state: AgentState) -> AgentState:
    
    print("=== ANSWER CRITIQUE ===")
    
    critic_llm = llm.with_structured_output(CritiqueFeedback)
    
    # Gather tool execution results for context
    tool_results = []
    for msg in state["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_results.extend([f"Tool: {tc['name']}, Args: {tc['args']}" for tc in msg.tool_calls])
    
    if state.get("plan"):
        terra = state.get("plan")
    else:
        terra = "No plan used"
    critique_prompt = CRITIC_PROMPT.format(
        query=state["query"],
        plan=terra,
        answer=state["final_answer"],
        tool_results=tool_results[:5]   #Limit context
    )
    
    critique = critic_llm.invoke([
        SystemMessage(content=critique_prompt),
        HumanMessage(content="Please evaluate this answer.")
    ])
    
    print(f"Quality Score: {critique.quality_score}/10")
    print(f"Complete: {critique.is_complete}")
    print(f"Accurate: {critique.is_accurate}")
    if critique.errors_found:
        print(f"Errors: {critique.errors_found}")
    if critique.needs_replanning:
        print(f"Needs replanning: {critique.replan_instructions}")
    
    return {
        "critique_feedback": critique,
        "iteration_count": state.get("iteration_count", 0) + 1
    }
"""

def critic_evaluator(state: AgentState) -> AgentState:
    """Enhanced critic that evaluates execution reports."""
    print("=== ENHANCED ANSWER CRITIQUE ===")
    
    report = state.get("execution_report")
    critic_llm = llm.with_structured_output(CritiqueFeedback)
    
    critique_prompt = CRITIC_PROMPT.format(
        query=report.query_summary,
        approach=report.approach_used,
        tools=report.tools_executed,
        findings=report.key_findings,
        sources=report.data_sources,
        confidence=report.confidence_level,
        limitations=report.limitations,
        answer=report.final_answer
    )
    
    critique = critic_llm.invoke([
        SystemMessage(content=critique_prompt),
        HumanMessage(content="Evaluate this execution report thoroughly.")
    ])
    
    print(f"Quality Score: {critique.quality_score}/10")
    print(f"Complete: {critique.is_complete}")
    print(f"Accurate: {critique.is_accurate}")
    
    if critique.errors_found:
        print(f"Issues found: {critique.errors_found}")
    
    if critique.needs_replanning:
        print(f"Replanning needed: {critique.replan_instructions}")
    
    return {
        "critique_feedback": critique,
        "iteration_count": state.get("iteration_count", 0) + 1
    }



def should_replan(state: AgentState) -> str:
    """Decide whether to accept answer, replan, or stop."""
    critique = state.get("critique_feedback")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    

    print(f"=== REPLAN DECISION ===")
    print(f"Iteration: {iteration_count}/{max_iterations}")
    print(f"Quality score: {critique.quality_score if critique else 'N/A'}")
    print(f"Needs replanning: {critique.needs_replanning if critique else 'N/A'}")

    if not critique:
        return "end"
    
    # Stop if max iterations reached
    if iteration_count >= max_iterations:
        print(f"Max iterations ({max_iterations}) reached. Accepting current answer.")
        return "end"
    
    # Accept if quality is good enough
    if critique.quality_score >= 7 or not critique.needs_replanning:
        print("Quality acceptable, ending execution")
        return "end"
    
    # Replan if quality is poor and we haven't exceeded max iterations
    if critique.needs_replanning and iteration_count < max_iterations:
        print("Replanning due to critic feedback...")
        return "replan"
    
    return "end"

def replanner(state: AgentState) -> AgentState:
    """Create a revised plan based on critic feedback."""
    print("=== REPLANNING ===")
    
    critique = state["critique_feedback"]
    previous_plan = state.get("plan")
    
    replan_prompt = f"""
    {SYSTEM_PROMPT_PLANNER}
    
    REPLANNING CONTEXT:
    Original Query: {state['query']}
    Previous Plan: {previous_plan if previous_plan else {}}
    
    CRITIC FEEDBACK:
    - Quality Score: {critique.quality_score}/10
    - Issues Found: {critique.errors_found}
    - Missing Elements: {critique.missing_elements}
    - Improvement Suggestions: {critique.suggested_improvements}
    - Specific Instructions: {critique.replan_instructions}
    
    Create a REVISED plan that addresses these issues. Focus on fixing the identified problems.
    """
    
    revised_plan = planner_llm.invoke([
        SystemMessage(content=replan_prompt),
        HumanMessage(content="Create a revised plan based on the feedback.")
    ])
    
    print("Plan revised based on critic feedback")
    
    # Очищаем историю сообщений от неполных tool_calls
    current_messages = state.get("messages", [])
    cleaned_messages = clean_message_history(current_messages)
    
    # Оставляем только системные сообщения и начальный запрос
    essential_messages = []
    for msg in cleaned_messages:
        if isinstance(msg, (SystemMessage, HumanMessage)):
            # Сохраняем системные сообщения и пользовательские запросы
            if ("complexity" in msg.content.lower() or 
                "assess" in msg.content.lower() or
                isinstance(msg, HumanMessage)):
                essential_messages.append(msg)
    
    print(f"Cleaned message history: {len(current_messages)} -> {len(essential_messages)} messages")
    
    return {
        "plan": revised_plan,
        "current_step": 0,
        "reasoning_done": False,
        "messages": essential_messages,
        "execution_report": None
    }

# === CELL 4 FROM NOTEBOOK ===
#GRAPH BUILDING

builder = StateGraph(AgentState)
builder.add_node("INPUT", query_input)
builder.add_node("COMPLEXITY_ASSESSOR", complexity_assessor)
builder.add_node("PLANNING", planner)
builder.add_node("AGENT", agent)
builder.add_node("TOOLS", DEBUGGING_TOOL_NODE)
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
builder.add_edge("SIMPLE_EXECUTOR", "FINALIZER")


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


system = builder.compile(checkpointer=MemorySaver())

# === CELL 5 FROM NOTEBOOK ===
workflow = system.invoke({"query" : "How many cumulative milliliters of fluid is in all the opaque-capped vials without stickers in the 114 version of the kit that was used for the PromethION long-read sequencing in the paper De Novo-Whole Genome Assembly of the Roborovski Dwarf Hamster (Phodopus roborovskii) Genome?", "current_step": 0, "reasoning_done": False, "files" : [], "files_contents" : {}, "iteration_count" : 0, "max_iterations" : 10, "plan" : None} , config = config)

# === CELL 6 FROM NOTEBOOK ===
for message in workflow["messages"]:
    message.pretty_print()

print("\n=== FINAL ANSWER ===")

# === CELL 7 FROM NOTEBOOK ===
workflow["final_answer"]

# === CELL 8 FROM NOTEBOOK ===
workflow

# === CELL 9 FROM NOTEBOOK ===
#TO-DO:
# - imrove image generation and plots/tables creation
# - add more tools (e.g. calendar, email, pdf editing, file system)
# - UI creation
