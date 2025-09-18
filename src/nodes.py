import os
from typing import Optional

from state import AgentState
from tools.tools import preprocess_files
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from prompts.prompts import (
    SYSTEM_PROMPT_PLANNER,
    SYSTEM_EXECUTOR_PROMPT,
    COMPLEXITY_ASSESSOR_PROMPT,
    CRITIC_PROMPT,
)
from config import llm, TOOLS, planner_llm, llm_with_tools
from schemas import PlannerPlan, ComplexityLevel, CritiqueFeedback, ExecutionReport, ToolExecution
from utils.utils import (
    format_final_answer,
    clean_message_history,
    log_stage,
    log_key_values,
    display_plan,
    format_plan_overview,
)


def _build_planner_prompt(state: AgentState, extra_context: Optional[str] = None) -> str:
    tool_catalogue = ", ".join(sorted(tool.name for tool in TOOLS))
    file_paths = state.get("files", [])
    file_list = ", ".join(os.path.basename(path) for path in file_paths) if file_paths else "none provided"
    extra = extra_context.strip() if extra_context else "None"
    return SYSTEM_PROMPT_PLANNER.format(
        tool_catalogue=tool_catalogue,
        file_list=file_list,
        extra_context=extra,
    ).strip()

def query_input(state: AgentState) -> AgentState:
    log_stage("USER QUERY", icon="💡")

    files = state.get("files", [])
    if files:
        log_stage("FILE PREPARATION", subtitle=f"Processing {len(files)} file(s)", icon="📁")
        file_info = preprocess_files(files)

        for file_path, info in file_info.items():
            log_key_values(
                [
                    ("path", file_path),
                    ("type", info["type"]),
                    ("size", f"{info['size']} bytes"),
                    ("suggested_tool", info["suggested_tool"]),
                ]
            )

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

        file_context += "IMPORTANT: Use the suggested tools to analyze these files before processing their data.\n"
        file_context += "File paths are available in the agent state and can be passed directly to analysis tools.\n"

        original_query = state.get("query", "")
        state["query"] = original_query + file_context
    else:
        log_key_values([("files", "none provided")])
    return state


def planner(state: AgentState) -> AgentState:
    log_stage("PLANNING", icon="🧭")
    planner_prompt = _build_planner_prompt(state)

    sys_stack = [
        SystemMessage(content=planner_prompt),
        HumanMessage(content=state["query"]),
    ]
    plan: PlannerPlan = planner_llm.invoke(sys_stack)

    display_plan(plan)
    return {
        "messages": state["messages"] + sys_stack,
        "plan": plan,
        "current_step": 0,
        "reasoning_done": False,
    }


def agent(state: AgentState) -> AgentState:
    current_step = state.get("current_step", 0)
    reasoning_done = state.get("reasoning_done", False)
    plan: Optional[PlannerPlan] = state.get("plan")

    if not plan or not hasattr(plan, "steps"):
        log_stage("PLAN VALIDATION", subtitle="Planner returned no actionable steps", icon="⚠️")
        warning = AIMessage(content="No valid plan available. <FINAL_ANSWER>")
        return {
            "messages": state["messages"] + [warning],
            "reasoning_done": False,
        }

    steps = plan.steps
    total_steps = len(steps)

    if total_steps == 0:
        log_stage("PLAN VALIDATION", subtitle="Plan indicates direct answer", icon="ℹ️")
        direct = AIMessage(content="Plan has no steps; respond directly. <FINAL_ANSWER>")
        return {
            "messages": state["messages"] + [direct],
            "reasoning_done": False,
        }

    if current_step >= total_steps:
        log_stage("PLAN COMPLETE", subtitle="All steps executed", icon="✅")
        completion = AIMessage(content="All plan steps completed. <FINAL_ANSWER>")
        return {
            "messages": state["messages"] + [completion],
            "reasoning_done": False,
        }

    current_step_info = steps[current_step]
    log_stage(
        "EXECUTION",
        subtitle=f"Step {current_step + 1}/{total_steps}: {current_step_info.goal}",
        icon="🤖",
    )
    log_key_values(
        [
            ("step_id", current_step_info.id),
            ("tool", current_step_info.tool or "none"),
            ("expected", current_step_info.expected_result),
        ]
    )

    plan_overview = format_plan_overview(plan)
    tool_catalogue = ", ".join(sorted(tool.name for tool in TOOLS))
    file_contents = state.get("file_contents", {})
    file_list = ", ".join(file_contents.keys()) if file_contents else "none provided"

    system_message = SystemMessage(
        content=SYSTEM_EXECUTOR_PROMPT.format(
            plan_summary=plan.summary,
            plan_overview=plan_overview,
            current_step_id=current_step_info.id,
            step_goal=current_step_info.goal,
            step_tool=current_step_info.tool or "no tool (respond directly)",
            tool_catalogue=tool_catalogue,
            file_list=file_list,
        ).strip()
    )

    if not reasoning_done:
        instruction = HumanMessage(
            content=(
                "Provide reasoning for this step inside <REASONING>...</REASONING>. "
                "Do not call any tools yet."
            )
        )
        stack = [system_message] + state["messages"] + [instruction]
        reasoning_response = llm.invoke(stack)
        log_stage("REASONING", subtitle=f"{current_step_info.id}", icon="🧠")
        print(reasoning_response.content)

        return {
            "messages": state["messages"] + [reasoning_response],
            "reasoning_done": True,
        }

    available_tools = {tool.name for tool in TOOLS}
    if current_step_info.tool and current_step_info.tool not in available_tools:
        log_stage(
            "TOOL WARNING",
            subtitle=f"Unknown tool '{current_step_info.tool}' in plan",
            icon="⚠️",
        )
        warning = AIMessage(
            content=(
                f"<REASONING>Unable to execute {current_step_info.id}: tool "
                f"'{current_step_info.tool}' is unavailable. Requesting replanning.</REASONING>"
            )
        )
        print(warning.content)
        return {
            "messages": state["messages"] + [warning],
            "reasoning_done": False,
        }

    execution_instruction = HumanMessage(
        content=(
            "Execute the planned action now. If a tool is required, call it with the "
            "correct arguments. After success, respond with STEP COMPLETE. If inputs are "
            "missing, explain the issue in <REASONING> without new tool calls."
        )
    )
    stack = [system_message] + state["messages"] + [execution_instruction]
    execution_response = llm_with_tools.invoke(stack)

    if execution_response.tool_calls:
        tool_names = ", ".join(call["name"] for call in execution_response.tool_calls)
        log_stage("TOOL CALL", subtitle=f"{current_step_info.id} → {tool_names}", icon="🛠️")
        print(execution_response.tool_calls)
    else:
        log_stage("EXECUTION OUTPUT", subtitle=current_step_info.id, icon="🛠️")
        if execution_response.content:
            print(execution_response.content)

    advance = False
    if execution_response.tool_calls:
        advance = True
    elif execution_response.content and (
        "STEP COMPLETE" in execution_response.content or "<FINAL_ANSWER>" in execution_response.content
    ):
        advance = True

    next_step = current_step + 1 if advance and current_step < total_steps else current_step

    return {
        "messages": state["messages"] + [execution_response],
        "current_step": next_step,
        "reasoning_done": False,
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
        log_stage("TOOL NODE", subtitle="Dispatching tool calls", icon="🛠️")
        try:
            result = super().__call__(state)
            log_stage("TOOL NODE", subtitle="Tool execution completed", icon="✅")
            return result
        except Exception as exc:
            log_stage("TOOL ERROR", subtitle=f"{type(exc).__name__}: {exc}", icon="❌")
            messages = state.get("messages", [])
            last_message = messages[-1] if messages else None
            tool_calls = getattr(last_message, "tool_calls", []) if last_message else []

            error_messages = []
            for call in tool_calls:
                error_messages.append(
                    ToolMessage(
                        content=f"ERROR: {type(exc).__name__}: {exc}",
                        tool_call_id=call.get("id") or "unknown_call",
                        name=call.get("name"),
                    )
                )

            if not error_messages:
                error_messages.append(
                    ToolMessage(
                        content=f"ERROR: {type(exc).__name__}: {exc}",
                        tool_call_id="unknown_call",
                    )
                )

            return {"messages": messages + error_messages}
    


def enhanced_finalizer(state: AgentState) -> AgentState:
    """Generate comprehensive execution report for critic evaluation."""
    log_stage("FINALIZER", subtitle="Compiling execution report", icon="📄")

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
    plan_overview = ""

    if plan:
        approach_used = f"{plan.task_type} plan – {plan.summary}"
        assumptions_made = plan.assumptions
        plan_overview = format_plan_overview(plan)
    
    # Generate structured report (КОСТЫЛЬ ЗДЕСЬ!)
    report_generator_prompt = f"""
    Generate a comprehensive execution report for the following query processing:

    ORIGINAL QUERY: {state['query']}

    EXECUTION CONTEXT:
    - Complexity Level: {state.get('complexity_assessment', {}).level}
    - Plan Used: {plan_overview if plan_overview else 'direct response'}
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
    
    log_key_values(
        [
            ("confidence", execution_report.confidence_level),
            ("findings", str(len(execution_report.key_findings))),
            ("sources", str(len(execution_report.data_sources))),
        ]
    )

    # Format final answer for user
    formatted_answer = format_final_answer(execution_report, state.get('complexity_assessment', {}))
    log_stage("FINAL ANSWER PREVIEW", icon="📬")
    print(formatted_answer)
    return {
        "execution_report": execution_report,
        "final_answer": formatted_answer
    }


def simple_executor(state: AgentState) -> AgentState:
    """Handle simple queries directly without planning."""
    log_stage("SIMPLE EXECUTION", subtitle="Handling low-complexity query", icon="⚡")

    # For simple queries, use the LLM with tools directly
    simple_prompt = f"""
    Answer this simple query directly and efficiently: {state['query']}

    Stay factual, cite tools only if you actually call them, and avoid inventing files or URLs.
    Known files: {list(state.get('file_contents', {}).keys())}
    If no tool is required, respond immediately with the final answer.
    """

    response = llm_with_tools.invoke([
        SystemMessage(content=simple_prompt),
        HumanMessage(content=state['query'])
    ])
    
    log_stage("SIMPLE EXECUTION OUTPUT", icon="📬")
    print(response.content)

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
    

def critic_evaluator(state: AgentState) -> AgentState:
    """Enhanced critic that evaluates execution reports."""
    log_stage("CRITIC", subtitle="Evaluating execution report", icon="🔍")

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
    
    log_key_values(
        [
            ("quality", f"{critique.quality_score}/10"),
            ("complete", str(critique.is_complete)),
            ("accurate", str(critique.is_accurate)),
        ]
    )

    if critique.errors_found:
        log_stage("CRITIC ISSUES", icon="⚠️")
        for issue in critique.errors_found:
            print(f" - {issue}")

    if critique.needs_replanning:
        log_stage("CRITIC REPLAN", subtitle="Replanning requested", icon="♻️")
        print(critique.replan_instructions)
    
    return {
        "critique_feedback": critique,
        "iteration_count": state.get("iteration_count", 0) + 1
    }



def should_replan(state: AgentState) -> str:
    """Decide whether to accept answer, replan, or stop."""
    critique = state.get("critique_feedback")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    subtitle = f"Iteration {iteration_count}/{max_iterations}"
    log_stage("REPLAN DECISION", subtitle=subtitle, icon="🧭")
    if critique:
        log_key_values(
            [
                ("quality", str(critique.quality_score)),
                ("needs_replanning", str(critique.needs_replanning)),
            ]
        )

    if not critique:
        return "end"

    # Stop if max iterations reached
    if iteration_count >= max_iterations:
        log_stage("REPLAN DECISION", subtitle="Max iterations reached", icon="🛑")
        return "end"

    # Accept if quality is good enough
    if critique.quality_score >= 7 or not critique.needs_replanning:
        log_stage("REPLAN DECISION", subtitle="Accepting current answer", icon="✅")
        return "end"

    # Replan if quality is poor and we haven't exceeded max iterations
    if critique.needs_replanning and iteration_count < max_iterations:
        log_stage("REPLAN DECISION", subtitle="Triggering replanner", icon="♻️")
        return "replan"

    return "end"

def replanner(state: AgentState) -> AgentState:
    """Create a revised plan based on critic feedback."""
    log_stage("REPLANNER", subtitle="Adjusting plan based on feedback", icon="♻️")

    critique = state["critique_feedback"]
    previous_plan = state.get("plan")

    previous_summary = previous_plan.summary if previous_plan else "no previous plan"
    issues = ", ".join(critique.errors_found) if critique.errors_found else "none"
    improvements = ", ".join(critique.suggested_improvements) if critique.suggested_improvements else "none"
    extra_context = (
        f"Replanning requested by critic. Previous plan summary: {previous_summary}. "
        f"Critic score: {critique.quality_score}/10. Issues: {issues}. "
        f"Improvements to address: {improvements}. Specific instructions: "
        f"{critique.replan_instructions or 'none'}"
    )

    replan_prompt = _build_planner_prompt(state, extra_context=extra_context)

    revised_plan = planner_llm.invoke([
        SystemMessage(content=replan_prompt),
        HumanMessage(content="Create a revised plan based on the feedback.")
    ])

    display_plan(revised_plan)

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
    
    log_stage(
        "REPLANNER",
        subtitle=f"Cleaned history: {len(current_messages)} → {len(essential_messages)}",
        icon="🧹",
    )

    return {
        "plan": revised_plan,
        "current_step": 0,
        "reasoning_done": False,
        "messages": essential_messages,
        "execution_report": None
    }


def complexity_assessor(state: AgentState) -> AgentState:
    """Assess query complexity and determine if planning is needed."""
    log_stage("COMPLEXITY", subtitle="Assessing task difficulty", icon="📊")

    complexity_llm = llm.with_structured_output(ComplexityLevel)

    assessment_message = [
        SystemMessage(content=COMPLEXITY_ASSESSOR_PROMPT.strip()),
        HumanMessage(content=f"Query: {state['query']}")
    ]

    assessment = complexity_llm.invoke(assessment_message)
    log_key_values(
        [
            ("level", assessment.level),
            ("needs_planning", str(assessment.needs_planning)),
            ("reasoning", assessment.reasoning),
        ]
    )

    return {
        "complexity_assessment": assessment,
        "messages": state["messages"] + assessment_message
    }
