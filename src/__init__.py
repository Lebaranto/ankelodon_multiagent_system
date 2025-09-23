"""ANKELODON: Core AI Agent Package.

Import key components for easy use:
from src import workflow, llm
"""

"""
Ankelodon Multi-Agent System – package init.

Экспортирует удобный публичный API для работы с графом, состоянием агента,
схемами и конфигом. Клади этот файл в директорию, где лежат:
agent.py, config.py, nodes.py, schemas.py, state.py
(у тебя это src/).
"""

# Версия пакета (по желанию обновляй вручную/из git)
__version__ = "0.1.0"

# ── Граф/сборка
from .agent import build_workflow

# ── Состояние
from .state import AgentState

# ── Схемы/модели
from .schemas import (
    ComplexityLevel,
    CritiqueFeedback,
    PlannerPlan,
    PlanStep,
    ExecutionReport,
    ToolExecution,
    TaskType,
)

# ── Конфиг/LLM/Tools
from .config import (
    config,
    TOOLS,
    DEBUGGING_TOOL_NODE,
    llm,
    llm_deterministic,
    planner_llm,
    llm_with_tools,
    llm_criticist,
    llm_reasoning,
)

# ── Узлы/роутеры (если нужно вызывать напрямую или для тестов)
from .nodes import (
    query_input,
    complexity_assessor,
    planner,
    agent,
    simple_executor,
    critic_evaluator,
    replanner,
    enhanced_finalizer,
    # роутеры
    should_continue,
    should_use_planning,
    should_replan,
    should_use_tools_simple_executor,
)

__all__ = [
    # версия
    "__version__",
    # сборка графа
    "build_workflow",
    # состояние
    "AgentState",
    # схемы
    "ComplexityLevel",
    "CritiqueFeedback",
    "PlannerPlan",
    "PlanStep",
    "ExecutionReport",
    "ToolExecution",
    "TaskType",
    # конфиг/модели/тулы
    "config",
    "TOOLS",
    "DEBUGGING_TOOL_NODE",
    "llm",
    "llm_deterministic",
    "planner_llm",
    "llm_with_tools",
    "llm_criticist",
    "llm_reasoning",
    # узлы и роутеры
    "query_input",
    "complexity_assessor",
    "planner",
    "agent",
    "simple_executor",
    "critic_evaluator",
    "replanner",
    "enhanced_finalizer",
    "should_continue",
    "should_use_planning",
    "should_replan",
    "should_use_tools_simple_executor",
]
