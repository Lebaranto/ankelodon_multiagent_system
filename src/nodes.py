import os
from state import AgentState
from tools.tools import preprocess_files
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from prompts.prompts import SYSTEM_PROMPT_PLANNER, SYSTEM_EXECUTOR_PROMPT, COMPLEXITY_ASSESSOR_PROMPT, CRITIC_PROMPT
from config import llm, TOOLS, planner_llm, llm_with_tools
from schemas import PlannerPlan, ComplexityLevel, CritiqueFeedback, ExecutionReport, ToolExecution
from utils.utils import format_final_answer, clean_message_history

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
