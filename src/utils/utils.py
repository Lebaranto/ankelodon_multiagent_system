from typing import Iterable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from schemas import ComplexityLevel, ExecutionReport, PlannerPlan
from prompts.prompts import COMPLEXITY_ASSESSOR_PROMPT
from config import llm
from state import AgentState

def log_stage(title: str, subtitle: Optional[str] = None, icon: str = "🚀") -> None:
    """Render a banner for the current execution stage."""

    title_line = f" {title.strip()} "
    border = icon + " " + "═" * max(len(title_line), 20)
    print(f"\n{border}\n{icon} {title_line}\n{icon} " + "═" * max(len(title_line), 20))
    if subtitle:
        print(f"{icon} {subtitle}")


def log_key_values(pairs: Iterable[tuple[str, str]]) -> None:
    """Pretty-print simple key/value diagnostics."""

    for key, value in pairs:
        print(f"   • {key}: {value}")


def format_plan_overview(plan: PlannerPlan) -> str:
    """Create a human-readable summary of plan steps."""

    if not plan or not plan.steps:
        return "(no steps – direct response)"

    lines = []
    for step in plan.steps:
        tool_hint = step.tool if step.tool else "no tool"
        lines.append(f"{step.id}: {step.goal} [{tool_hint}]")
    return "\n".join(lines)


def display_plan(plan: PlannerPlan) -> None:
    """Print plan contents in a compact, readable form."""

    log_stage("PLANNER OUTPUT", icon="🧭")
    print(f"Task type: {plan.task_type}")
    print(f"Summary: {plan.summary}")
    if plan.assumptions:
        print("Assumptions:")
        for item in plan.assumptions:
            print(f"   - {item}")
    print("Steps:")
    for step in plan.steps:
        print(f"   {step.id} → {step.goal}")
        if step.tool:
            print(f"      tool: {step.tool}")
        if step.inputs:
            print(f"      inputs: {step.inputs}")
        print(f"      expected: {step.expected_result}")
        if step.on_fail:
            print(f"      on_fail: {step.on_fail}")
    if plan.answer_guidelines:
        print(f"Answer guidelines: {plan.answer_guidelines}")


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