from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict


@dataclass
class EvidenceChunk:
    """Represents a discrete retrieved text or table chunk with immutable metadata."""
    content: str
    source_id: str
    filename: str = "Unknown"
    page_number: Optional[int] = None
    content_type: str = "text"
    table_header_markdown: Optional[str] = None
    table_index: Optional[int] = None
    rerank_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "content": self.content,
            "source_id": self.source_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "content_type": self.content_type,
            "table_header_markdown": self.table_header_markdown,
            "table_index": self.table_index,
            "rerank_score": self.rerank_score,
            "hybrid_score": self.hybrid_score,
        }
        if self.extra_metadata:
            data.update(self.extra_metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceChunk:
        known_keys = {
            "content", "source_id", "filename", "page_number",
            "content_type", "table_header_markdown", "table_index",
            "rerank_score", "hybrid_score"
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            content=data.get("content", ""),
            source_id=data.get("source_id", ""),
            filename=data.get("filename", "Unknown"),
            page_number=data.get("page_number"),
            content_type=data.get("content_type", "text"),
            table_header_markdown=data.get("table_header_markdown"),
            table_index=data.get("table_index"),
            rerank_score=data.get("rerank_score"),
            hybrid_score=data.get("hybrid_score"),
            extra_metadata=extra,
        )


@dataclass
class AgentQueryRequest:
    """Request contract passed into the query workflow."""
    query: str
    chat_history: List[Tuple[str, str]] = field(default_factory=list)
    document_scope: Optional[str] = None
    top_k: int = 10
    session_id: Optional[str] = None


@dataclass
class PlannerOutput:
    """Output contract produced by QueryPlanner."""
    intent: str  # "full_doc" | "section" | "standard"
    target: Optional[str] = None
    retrieval_query: str = ""
    start_page: Optional[int] = None
    end_page: Optional[int] = None


@dataclass
class ValidationResult:
    """Citation validation assessment result."""
    is_valid: bool
    normalized_answer: str
    error_message: Optional[str] = None
    repair_count: int = 0


@dataclass
class AgentQueryResult:
    """Final output contract returned to UI or caller."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    citation_error: Optional[str] = None
    is_table_result: bool = False
    trace: Dict[str, Any] = field(default_factory=dict)


class WorkflowState(TypedDict, total=False):
    """LangGraph typed state dictionary."""
    # Request inputs
    query: str
    chat_history: List[Tuple[str, str]]
    document_scope: Optional[str]
    top_k: int
    session_id: Optional[str]

    # Planner
    planner_intent: str
    planner_target: Optional[str]
    retrieval_query: str

    # Evidence
    evidence_chunks: List[Dict[str, Any]]
    context_text: str

    # Intermediate calculation / generation
    table_answer: Optional[str]
    draft_answer: Optional[str]
    current_answer: Optional[str]

    # Citation & validation
    citation_error: Optional[str]
    repair_count: int
    expansion_count: int
    verified_answer: Optional[str]

    # Streaming flag
    is_streaming: bool

    # Trace & stats
    trace_steps: List[str]
    trace_details: Dict[str, Any]
    error: Optional[str]

