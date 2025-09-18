from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

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

class PlanStep(BaseModel):
    id: str = Field(description="Unique step identifier (e.g., s1)")
    goal: str = Field(description="What the step accomplishes and why")
    tool: Optional[str] = Field(default=None, description="Exact tool name or null when no tool is required")
    inputs: Optional[str] = Field(default=None, description="Important inputs or references needed for the step")
    expected_result: str = Field(description="How to confirm the step succeeded")
    on_fail: str = Field(default="replan", description="Fallback action if the step fails (replan or stop)")

    @field_validator("tool", mode="before")
    @classmethod
    def normalize_tool(cls, value: Optional[str]) -> Optional[str]:
        """Ensure blank or null-like values are interpreted as no tool."""

        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in {"null", "none"}:
                return None
            return cleaned
        return value

class PlannerPlan(BaseModel):
    task_type: TaskType
    summary: str = Field(description="Short explanation of the chosen strategy")
    assumptions: List[str] = Field(default_factory=list)
    steps: List[PlanStep] = Field(default_factory=list)
    answer_guidelines: Optional[str] = Field(default=None, description="Reminders for formatting, citations, etc.")


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