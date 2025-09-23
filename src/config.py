from langchain_openai import ChatOpenAI
from tools.tools import *
from tools.code_interpreter import safe_code_run
from langgraph.prebuilt import ToolNode
from schemas import PlannerPlan
from utils.utils import log_stage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from tools.youtube_transcript import extract_youtube_transcript
from dotenv import load_dotenv
load_dotenv()

config = {"configurable": {"thread_id": "1"}, "recursion_limit" : 50}

TOOLS = [download_file_from_url, web_search, 
         arxiv_search, wiki_search, add, subtract, multiply, divide, 
         power, analyze_excel_file, analyze_csv_file, analyze_docx_file, 
         analyze_pdf_file, analyze_txt_file, 
         vision_qa_gemma, safe_code_run, web_extract,extract_youtube_transcript]


TOOL_NODE = ToolNode(TOOLS)
DEBUGGING_TOOL_NODE = TOOL_NODE

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) #default 0.25
llm_deterministic = ChatOpenAI(model="gpt-5-mini", temperature=0.05)
planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).with_structured_output(PlannerPlan)
llm_criticist = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_with_tools = llm_deterministic.bind_tools(TOOLS)
llm_reasoning = ChatOpenAI(model="gpt-5-mini", temperature=0.3)
llm_simple_executor = ChatOpenAI(model="gpt-5-mini", temperature=0.3)
llm_simple_with_tools = llm_simple_executor.bind_tools(TOOLS)
finalizer_llm = ChatOpenAI(model="gpt-5-mini", temperature=0.5)



