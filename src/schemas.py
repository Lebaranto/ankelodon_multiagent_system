from typing import Any, Dict, List, Optional, Literal, Iterable
from pydantic import BaseModel, Field, ValidationError


class ComplexityLevel(BaseModel):
    level: Literal["simple", "moderate", "complex"] = Field(description="Complexity level of the query")
    reasoning: str = Field(description="Explanation for the complexity assessment")
    needs_planning: bool = Field(description="Whether this query requires detailed planning")
    suggested_approach: str = Field(description="Recommended approach for handling this query")

class CritiqueFeedback(BaseModel):
    quality_score: int = Field(ge=1, le=10, description="Quality score from 1-10")
    is_complete: bool = Field(description="Whether the answer is complete")
    is_accurate: bool = Field(description="Whether the answer appears accurate")
    missing_elements: List[str] = Field(default_factory=list, description="What's missing from the answer")
    errors_found: List[str] = Field(default_factory=list, description="Potential errors identified")
    suggested_improvements: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    needs_replanning: bool = Field(description="Whether the plan should be revised")
    replan_instructions: Optional[str] = Field(default=None, description="Instructions for replanning")



TaskType = Literal["info", "calc", "table", "doc_qa", "image_qa", "multi_hop"]
EvidenceTag = Literal["citations", "page_numbers", "figure_captions", "stats_check", "unit_check"]

class PlanStep(BaseModel):
    id: str
    description: str
    #tool: Optional[str] = Field(default=None, description="Exact tool name or null for reasoning step")
    #args_hint: Dict[str, Any] = Field(default_factory=dict)
    evidence_needed: List[EvidenceTag] = Field(default_factory=list)
    success_criteria: str
    on_fail: str = Field(default="replan", description="One of: 'replan' | 'stop' | step-id")
    outputs_to_state: List[str] = Field(default_factory=list)

class AnswerGuidelines(BaseModel):
    final_answer_template: str
    citations_required: bool = False
    min_citations: int = 0
    units_policy: Optional[str] = None
    rounding_policy: Optional[str] = None
    include_artifacts: List[str] = Field(default_factory=list)

class PlannerPlan(BaseModel):
    task_type: TaskType
    assumptions: List[str] = Field(default_factory=list)
    plan_rationale: str
    steps: List[PlanStep]
    answer_guidelines: AnswerGuidelines


class ToolExecution(BaseModel):
    tool_name: str
    arguments: str
    call_id: str
    
    class Config:
        extra = "forbid"

class ExecutionReport(BaseModel):
    """Structured report for critic evaluation."""
    query_summary: str = Field(description="Brief summary of the user's query")
    approach_used: str = Field(description="What approach/strategy was used")
    tools_executed: List[ToolExecution] = Field(default_factory=list, description="List of tools used with results")
    key_findings: List[str] = Field(default_factory=list, description="Main findings or results")
    data_sources: List[str] = Field(default_factory=list, description="Sources of information used")
    assumptions_made: List[str] = Field(default_factory=list, description="Any assumptions made during execution")
    confidence_level: Literal["low", "medium", "high"] = Field(description="Confidence in the answer")
    limitations: List[str] = Field(default_factory=list, description="Known limitations or caveats")
    final_answer: str = Field(description="The actual answer to the user's query")

    class Config:
        extra = "forbid"