"""
src/agents/summary_agent.py

Document Summary Agent — specialized agent for autonomous whole-document summarization.
Overcomes standard Top-K RAG limitations by extracting structural outlines, page-aware milestones,
and assembling multi-phase evidence spanning the entire document from page 1 to the end.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocumentSummaryAgent:
    """
    Autonomous whole-document summarization specialist.

    Solves the 'lost in the middle' and top-k chunk coverage limitations of Simple RAG by:
    1. Inspecting the document's total page count and structural section outline.
    2. Dynamically retrieving strategic multi-page anchors:
       - Preamble, Title & TOC (pages 1-3)
       - Body milestone chunks distributed across the document's page percentiles
       - Concluding & operational summary pages (tail)
    3. Assembling an ordered, deduplicated evidence package for executive synthesis.
    """

    def __init__(
        self,
        vectorstore: Any,
        llm_invoker: Optional[Callable[[list], str]] = None,
    ):
        self.vectorstore = vectorstore
        self.llm_invoker = llm_invoker

    def gather_full_document_evidence(
        self,
        filename: str,
        query: str = "",
        max_total_chunks: int = 18,
    ) -> List[Dict[str, Any]]:
        """
        Gather comprehensive evidence across the entire document lifecycle.
        """
        if not self.vectorstore or not filename:
            return []

        # 1. First attempt: Use vectorstore's get_document_outline if available
        if hasattr(self.vectorstore, "get_document_outline"):
            outline = self.vectorstore.get_document_outline(filename)
            outline_chunks = outline.get("outline_chunks", [])
            if outline_chunks:
                logger.info(
                    "[DocumentSummaryAgent] Assembled %d outline chunks for %s (pages: %d, sections: %d)",
                    len(outline_chunks),
                    filename,
                    outline.get("page_count", 1),
                    len(outline.get("sections", [])),
                )
                return outline_chunks[:max_total_chunks]

        # 2. Fallback / Test Doubles: Call query_by_pages across available range
        if hasattr(self.vectorstore, "query_by_pages"):
            try:
                chunks = self.vectorstore.query_by_pages(
                    filename=filename,
                    start_page=1,
                    end_page=500,
                    max_chunks=max_total_chunks,
                )
                if chunks:
                    return chunks
            except Exception as e:
                logger.debug("query_by_pages fallback error: %s", e)

        # 3. Final fallback: Standard semantic search
        if hasattr(self.vectorstore, "query"):
            return self.vectorstore.query(
                query_text=query or filename,
                filename=filename,
                top_k=max_total_chunks,
            )

        return []

