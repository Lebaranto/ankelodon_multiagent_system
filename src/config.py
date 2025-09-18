from langchain_openai import ChatOpenAI
from tools.tools import *
from tools.code_interpreter import safe_code_run
from langgraph.prebuilt import ToolNode
from schemas import PlannerPlan
from utils.utils import log_stage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

config = {"configurable": {"thread_id": "1"}, "recursion_limit" : 50}

TOOLS = [download_file_from_url, web_search, 
         arxiv_search, wiki_search, add, subtract, multiply, divide, 
         power, analyze_excel_file, analyze_csv_file, analyze_docx_file, 
         analyze_pdf_file, analyze_txt_file, analyze_image_file, 
         vision_qa_gemma, safe_code_run]

class DebuggingToolNode(ToolNode):
    def __init__(self, tools):
        super().__init__(tools)

    def __call__(self, state):
        log_stage("TOOL NODE", subtitle="Dispatching tool calls", icon="🛠️")
        
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if not last_message or not hasattr(last_message, "tool_calls"):
            log_stage("TOOL ERROR", subtitle="No tool calls found", icon="❌")
            return state
            
        tool_calls = last_message.tool_calls
        log_stage("TOOL DISPATCH", subtitle=f"Executing {len(tool_calls)} tool(s)", icon="🔧")
        for call in tool_calls:
            print(f"   - {call['name']}: {call['args']}")
        
        try:
            # Выполняем инструменты
            result = super().__call__(state)
            
            # Проверяем результаты
            new_messages = result.get("messages", [])
            tool_messages = [msg for msg in new_messages[len(messages):] 
                           if isinstance(msg, ToolMessage)]
            
            log_stage("TOOL RESULTS", subtitle=f"Got {len(tool_messages)} responses", icon="📨")
            
            # Логируем результаты
            for msg in tool_messages:
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"   - {msg.name}: {content_preview}")
            
            # Автоматически добавляем сигнал завершения шага после успешного выполнения инструментов
            if tool_messages:
                current_step = state.get("current_step", 0)
                plan = state.get("plan")
                
                if plan and current_step < len(plan.steps):
                    step_completion_msg = AIMessage(
                        content=f"STEP COMPLETE: Successfully executed {len(tool_messages)} tool(s) for step {plan.steps[current_step].id}"
                    )
                    result["messages"] = result["messages"] + [step_completion_msg]
                    log_stage("STEP COMPLETION", subtitle=f"Step {current_step + 1} marked complete", icon="✅")
                    
                    # Продвигаем к следующему шагу
                    result["current_step"] = current_step + 1
                    result["reasoning_done"] = False  # Сброс для следующего шага
            
            return result
            
        except Exception as exc:
            log_stage("TOOL ERROR", subtitle=f"{type(exc).__name__}: {exc}", icon="❌")
            print(f"Full error: {repr(exc)}")
            
            # Создаем ToolMessage для каждого failed tool call
            error_messages = []
            for call in tool_calls:
                error_msg = ToolMessage(
                    content=f"ERROR: {type(exc).__name__}: {exc}",
                    tool_call_id=call.get("id") or "unknown_call",
                    name=call.get("name", "unknown_tool"),
                )
                error_messages.append(error_msg)

            return {"messages": messages + error_messages}


TOOL_NODE = ToolNode(TOOLS)
DEBUGGING_TOOL_NODE = DebuggingToolNode(TOOLS)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.25)
llm_with_tools = llm.bind_tools(TOOLS)
planner_llm = llm.with_structured_output(PlannerPlan)




