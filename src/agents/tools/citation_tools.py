from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from src.services.citation_service import CitationService


def create_citation_tools(citation_service: Optional[CitationService] = None) -> List[Any]:
    """
    Creates and returns standard LangChain @tool definitions exposed
    specifically to the CitationAgent for verification and reference formatting.
    """
    cit = citation_service or CitationService()

    @tool
    def verify_citations(
        draft: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify that all [S<number>] citation tags in a draft answer match
        valid retrieved source chunks.

        Args:
            draft: The draft response text.
            evidence_chunks: Retrieved evidence chunks with source_id attributes.
        """
        normalized = cit.normalize_citations(draft, evidence_chunks)
        error = cit.citation_validation_error(normalized, evidence_chunks)
        return {
            "is_valid": error is None,
            "error": error,
            "normalized_draft": normalized,
        }

    @tool
    def format_source_references(
        evidence_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a Markdown formatted bibliography of all cited sources with filenames and pages.

        Args:
            evidence_chunks: List of retrieved evidence chunks.
        """
        return cit.format_source_references(evidence_chunks)

    @tool
    def extract_cited_fallback(
        evidence_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Extract direct grounded excerpt quotes when synthesis model repeatedly omits citations.

        Args:
            evidence_chunks: List of retrieved evidence chunks.
        """
        return cit.cited_excerpt_fallback(evidence_chunks)

    return [verify_citations, format_source_references, extract_cited_fallback]

