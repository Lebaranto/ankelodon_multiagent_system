from langgraph.graph import MessagesState
from typing import List, Annotated, Optional, Dict, Any
from schemas import PlannerPlan, ComplexityLevel, CritiqueFeedback, ExecutionReport
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

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
    previous_tool_results: Dict[str, str]  # НОВОЕ ПОЛЕ для сохранения результатов
    previous_final_answer: str  # НОВОЕ ПОЛЕ для сохранения предыдущих окончательных ответов

