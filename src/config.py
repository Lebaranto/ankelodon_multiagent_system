from langchain_openai import ChatOpenAI
from tools.tools import *
from tools.code_interpreter import safe_code_run
from langgraph.prebuilt import ToolNode
from schemas import PlannerPlan

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
        print("=== TOOL EXECUTION STARTED ===")
        result = super().__call__(state)
        print("=== TOOL EXECUTION COMPLETED ===")
        return result


TOOL_NODE = ToolNode(TOOLS)
DEBUGGING_TOOL_NODE = DebuggingToolNode(TOOLS)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.25)
llm_with_tools = llm.bind_tools(TOOLS)
planner_llm = llm.with_structured_output(PlannerPlan)




