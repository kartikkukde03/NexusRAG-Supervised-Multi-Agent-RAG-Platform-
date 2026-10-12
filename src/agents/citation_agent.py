from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.tools.citation_tools import create_citation_tools
from src.services.citation_service import CitationService

logger = logging.getLogger(__name__)


class CitationAgent:
    """
    Auditor and self-healing repair specialist equipped with exposed citation tools.
    Verifies citation integrity, executes bounded repair prompt if missing, and formats bibliography.
    """

    def __init__(
        self,
        llm_invoker: Optional[Callable[[list], str]] = None,
        citation_service: Optional[CitationService] = None,
    ):
        self.llm_invoker = llm_invoker
        self.citation_service = citation_service or CitationService()
        # Expose dedicated citation tools directly to this agent
        self.tools = create_citation_tools(self.citation_service)
        self._tool_map = {t.name: t for t in self.tools}

    def audit(
        self,
        draft: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate draft using the exposed verify_citations tool.
        """
        verify_tool = self._tool_map.get("verify_citations")
        if verify_tool:
            return verify_tool.invoke({
                "draft": draft,
                "evidence_chunks": evidence_chunks,
            })
        normalized = self.citation_service.normalize_citations(draft, evidence_chunks)
        error = self.citation_service.citation_validation_error(normalized, evidence_chunks)
        return {
            "is_valid": error is None,
            "error": error,
            "normalized_draft": normalized,
        }

    def format_bibliography(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        """Format final bibliography using exposed tool."""
        bib_tool = self._tool_map.get("format_source_references")
        if bib_tool:
            return bib_tool.invoke({"evidence_chunks": evidence_chunks})
        return self.citation_service.format_source_references(evidence_chunks)

    def extract_fallback(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        """Extract fallback using exposed tool."""
        fb_tool = self._tool_map.get("extract_cited_fallback")
        if fb_tool:
            return fb_tool.invoke({"evidence_chunks": evidence_chunks})
        return self.citation_service.cited_excerpt_fallback(evidence_chunks)

    def repair_answer(
        self,
        messages: List[Any],
        draft: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """Bounded repair prompt requesting a single retry with explicit citation constraints."""
        if not self.llm_invoker:
            return draft

        evidence_prompt = messages[0].content
        retry_messages = [
            SystemMessage(
                content=(
                    f"{evidence_prompt}\n\n"
                    "Citation repair is required. Answer the user's question "
                    "concisely using only the evidence above. Every factual "
                    "sentence must end with an exact source citation copied "
                    "from the evidence labels, such as [S1]. Do not use "
                    "parenthetical citations or citation formats other than "
                    "[S<number>]. Do not invent source IDs. If the evidence "
                    "does not support a complete answer, state that limitation "
                    "and cite the relevant excerpt."
                )
            ),
            *messages[1:],
            AIMessage(content=draft),
            HumanMessage(
                content=(
                    "Rewrite your answer now. Include the source citation "
                    "in the required [S1] format."
                )
            ),
        ]
        try:
            repaired = self.llm_invoker(retry_messages)
            return repaired
        except Exception as exc:
            logger.warning(f"[CitationAgent] Repair invocation failed: {exc}")
            return draft


# Backward-compatible alias
CitationRepairAgent = CitationAgent

