"""
src/agents/__init__.py

Public surface for the agents package.
All five plan-specified agents + their contracts are exported here.
EvidenceSufficiencyAnalyst is re-exported from research_agent (where it lives as a companion class).
"""

from src.agents.contracts import (
    AgentQueryRequest,
    AgentQueryResult,
    EvidenceChunk,
    PlannerOutput,
    ValidationResult,
    WorkflowState,
)
from src.agents.planner_agent import PlannerAgent, QueryPlanner
from src.agents.research_agent import ResearchAgent, EvidenceSufficiencyAnalyst
from src.agents.table_agent import TableAgent
from src.agents.writer_agent import WriterAgent, AnswerWriterAgent
from src.agents.citation_agent import CitationAgent, CitationRepairAgent

__all__ = [
    # Contracts
    "WorkflowState",
    "EvidenceChunk",
    "AgentQueryRequest",
    "AgentQueryResult",
    "PlannerOutput",
    "ValidationResult",
    # Agents (plan-specified)
    "PlannerAgent",
    "ResearchAgent",
    "TableAgent",
    "WriterAgent",
    "CitationAgent",
    # Backward-compatible aliases
    "QueryPlanner",
    "AnswerWriterAgent",
    "CitationRepairAgent",
    "EvidenceSufficiencyAnalyst",
]
