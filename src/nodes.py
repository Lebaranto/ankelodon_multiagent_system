import os
from state import AgentState
from tools.tools import preprocess_files
from typing import Optional
from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from prompts.prompts import (
    SYSTEM_PROMPT_PLANNER,
    SYSTEM_EXECUTOR_PROMPT,
    COMPLEXITY_ASSESSOR_PROMPT,
    CRITIC_PROMPT,
)

from config import llm_reasoning, TOOLS, planner_llm, llm_with_tools, llm_deterministic, llm_criticist, llm_simple_executor, llm_simple_with_tools, finalizer_llm
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

def query_input(state : AgentState) -> AgentState:
    log_stage("USER QUERY", icon="💡")

    files = state.get("files", [])
    if files:
        log_stage("FILE PREPARATION", subtitle=f"Processing {len(files)} file(s)", icon="📁")
        file_info = preprocess_files(files)
    
        for file_path, info in file_info.items():
            #print(f"  - {file_path}: {info['type']} ({info['size']} bytes) -> {info['suggested_tool']}")
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
        
        # Добавляем инструкции по работе с файлами
        file_context += "IMPORTANT: Use the suggested tools to analyze these files before processing their data.\n"
        file_context += "File paths are available in the agent state and can be passed directly to analysis tools.\n"
    
    else:
        log_key_values([("files", "none provided")])
        file_context = ""
        original_query = state.get("query", "")
        state["query"] = original_query + file_context
    return state


def planner(state : AgentState) -> AgentState:

    log_stage("PLANNING", icon="🧭")
    planner_prompt = _build_planner_prompt(state)

    sys_stack = [
            SystemMessage(content=planner_prompt),
            HumanMessage(content=state["query"]),
        ]
    plan: PlannerPlan = planner_llm.invoke(sys_stack)
    
    #print("=== GENERATED PLAN ===")
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
    previous_tool_results = state.get("previous_tool_results", {})

    #steps = state["plan"].steps

    if not plan or not hasattr(plan, 'steps'):
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

    # Добавляем информацию о предыдущих результатах (UPDATE)
    previous_results_context = ""
    if previous_tool_results:
        previous_results_context = f"\n\nPREVIOUS CALCULATION RESULTS:\n"
        for tool_call_id, result in previous_tool_results.items():
            previous_results_context += f"- {tool_call_id}: {result}\n"
        previous_results_context += "You can reference these results in your calculations.\n"


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

        log_stage("REASONING", subtitle=f"{current_step_info.id}", icon="🧠")
        #print(reasoning_response.content)

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
        FOR MORE COMPLEX UNDERSTANDING -> USE RESULTS AND INSIGHTS FROM PREVIOUS STEPS.
        """

        sys_msg = SystemMessage(content = reasoning_prompt)
        stack = [sys_msg] + state["messages"]

        step = llm_reasoning.invoke(stack)
        #print("=== REASONING STEP ===")
        #print(step.content)

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
        #print(step)
        print(f"Tool calls: {step.tool_calls}")
        
        return {
            "messages": state["messages"] + [step],
            "current_step": current_step + 1 if step.tool_calls else current_step,
            "reasoning_done": False  # Сбрасываем для следующего шага
        }
    
def should_continue(state : AgentState) -> bool:
    
    last_message = state["messages"][-1]
    #print(f"=== LAST MESSAGE WAS: {last_message} ===")
    reasoning_done = state.get("reasoning_done", False)
    plan = state.get("plan", None)
    current_step = state.get("current_step", 0)

    print(f"=== SHOULD_CONTINUE DEBUG ===")
    print(f"Current step: {current_step}")
    print(f"Plan steps: {len(plan.steps) if plan else 0}")
    print(f"Reasoning done: {reasoning_done}")
    print(f"Last message type: {type(last_message).__name__}")

    #ПРИОРИТЕТ 1: Если есть tool_calls - выполняем их
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # ПРИОРИТЕТ 2: Явный сигнал завершения
    if hasattr(last_message, "content") and "<FINAL_ANSWER>" in last_message.content:
        return "final_answer"
    
    # ПРИОРИТЕТ 3: Логика reasoning/execution
    if not reasoning_done and hasattr(last_message, 'content') and "<REASONING>" in last_message.content:
        # Reasoning выполнен, но инструменты еще не вызваны
        return "agent"
    elif reasoning_done:
        # Reasoning выполнен, теперь нужно вызвать инструменты
        return "agent"
    elif not reasoning_done:
        # Нужно сделать reasoning
        return "agent"
    
    # ПРИОРИТЕТ 4: Проверяем завершение плана (только если нет активных tool_calls)
    if plan and current_step >= len(plan.steps):
        return "final_answer"
    
    # По умолчанию продолжаем выполнение
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
    
    report_llm = finalizer_llm.with_structured_output(ExecutionReport) #default llm_deterministic
    
    execution_report = report_llm.invoke([
        SystemMessage(content=report_generator_prompt),
        HumanMessage(content="Generate the execution report.")
    ])
    
    print(f"Report generated - Confidence: {execution_report.confidence_level}")
    print(f"Key findings: {len(execution_report.key_findings)}")
    print(f"Data sources: {len(execution_report.data_sources)}")
    
    # Format final answer for user
    formatted_answer = format_final_answer(execution_report, state.get('complexity_assessment', {}))
    #print(execution_report)
    print(f"FINAL ANSWER FOR EVALUATOR: {execution_report.final_answer}")
    return {
        "execution_report": execution_report,
        "final_answer": formatted_answer
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
    
    response = llm_simple_with_tools.invoke([
        SystemMessage(content=simple_prompt),
        HumanMessage(content=state['query'])
    ])

    print("Response generated for simple query.")
    
    return {
        "messages": state["messages"] + [response],
        "final_answer": response.content
    }

def should_use_tools_simple_executor(state: AgentState) -> str:
    """Decide whether to use tools or answer directly in simple executor."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    if hasattr(last_message, "content") and "<FINAL_ANSWER>" in last_message.content:
        return "final_answer"
    
    return "final_answer"


def should_use_planning(state: AgentState) -> str:
    """Route based on complexity assessment."""
    complexity = state["complexity_assessment"]
    
    if complexity.level == "simple" and not complexity.needs_planning:
        return "simple_executor"
    else:
        return "planner"
    

def critic_evaluator(state: AgentState) -> AgentState:
    """Enhanced critic that evaluates execution reports."""
    print("=== ENHANCED ANSWER CRITIQUE ===")
    
    report = state.get("execution_report")
    critic_llm = llm_criticist.with_structured_output(CritiqueFeedback)
    
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
    activator = state.get("critic_replan", False)

    print(f"=== REPLAN DECISION ===")
    print(f"Iteration: {iteration_count}/{max_iterations}")
    print(f"Quality score: {critique.quality_score if critique else 'N/A'}")
    print(f"Needs replanning: {critique.needs_replanning if critique else 'N/A'}")

    #для деактивации линии перепланировки, если это не нужно
    if not activator:
        return "end"

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

def replanner_old(state: AgentState) -> AgentState:
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
    
    #print(f"Cleaned message history: {len(current_messages)} -> {len(essential_messages)} messages")
    #print("=== ESSENTIAL MESSAGES ===")
    #print(essential_messages)
    #print("=== AGENT STATE ===")
    #print(state["messages"])

    return {
        "plan": revised_plan,
        "current_step": 0,
        "reasoning_done": False,
        "messages": essential_messages,
        "execution_report": None
    }

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
    
    # ИСПРАВЛЕНИЕ: Сохраняем важные результаты инструментов
    current_messages = state.get("messages", [])
    state["previous_final_answer"] = state.get("final_answer", "")
    # Находим полезные результаты инструментов
    preserved_messages = []
    tool_results = {}
    
    for i, msg in enumerate(current_messages):
        # Сохраняем системные сообщения и пользовательские запросы
        if isinstance(msg, (SystemMessage, HumanMessage)):
            # Фильтруем только исходные запросы, не промпты планировщика
            if (isinstance(msg, HumanMessage) or 
                ("complexity" in msg.content.lower() and "assessor" in msg.content.lower())):
                preserved_messages.append(msg)
        
        # Сохраняем успешные результаты инструментов
        elif isinstance(msg, ToolMessage) and msg.content and msg.content.strip():
            # Проверяем, что это полезный результат
            try:
                # Если результат можно преобразовать в число - это вычисление
                float(msg.content.strip())
                preserved_messages.append(msg)
                tool_results[msg.tool_call_id] = msg.content
                
                # Также нужно сохранить соответствующий AIMessage с tool_call
                for j in range(i-1, -1, -1):
                    if (isinstance(current_messages[j], AIMessage) and 
                        hasattr(current_messages[j], 'tool_calls') and
                        current_messages[j].tool_calls):
                        for tool_call in current_messages[j].tool_calls:
                            if tool_call['id'] == msg.tool_call_id:
                                if current_messages[j] not in preserved_messages:
                                    preserved_messages.insert(-1, current_messages[j])
                                break
                        break
            except (ValueError, AttributeError):
                # Если не число, но содержательный результат, тоже сохраняем
                if len(msg.content.strip()) > 1: # Минимальная длина для сохранения
                    preserved_messages.append(msg)
    
    print(f"Preserved {len(tool_results)} tool results")
    #print(f"Cleaned message history: {len(current_messages)} -> {len(preserved_messages)} messages")
    
    # Добавляем контекст о доступных результатах
    if tool_results:
        context_msg = HumanMessage(
            content=f"Previous calculation results available: {tool_results}"
        )
        preserved_messages.append(context_msg)

    return {
        "plan": revised_plan,
        "current_step": 0,
        "reasoning_done": False,
        "messages": preserved_messages,
        "execution_report": None,
        # Сохраняем важную информацию о предыдущих вычислениях
        "previous_tool_results": tool_results
    }

def complexity_assessor(state: AgentState) -> AgentState:
    """Assess query complexity and determine if planning is needed."""
    print("=== COMPLEXITY ASSESSMENT ===")
    
    complexity_llm = llm_deterministic.with_structured_output(ComplexityLevel)
    
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