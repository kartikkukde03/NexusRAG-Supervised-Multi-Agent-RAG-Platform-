from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class CitationService:
    """Deterministic service for normalizing citations, validating source references, and generating fallbacks."""

    @staticmethod
    def normalize_citations(answer: str, results: List[Dict[str, Any]]) -> str:
        """Standardize various citation formats (e.g. [Source 1], [1], (S1)) into [S1]."""
        normalized = answer
        for i in range(1, len(results) + 1):
            target = f"[S{i}]"
            # Replace [Source 1], [source 1], [Doc 1], [Document 1]
            normalized = re.sub(
                rf"\[(?:Source|source|Doc|Document)\s*{i}\]",
                target,
                normalized,
            )
            # Replace (S1), (s1)
            normalized = re.sub(rf"\([Ss]{i}\)", target, normalized)
            # Replace [1] if only bracketed numbers are used
            normalized = re.sub(rf"\[{i}\]", target, normalized)
        return normalized

    @staticmethod
    def citation_validation_error(answer: str, results: List[Dict[str, Any]]) -> Optional[str]:
        """Validate that answer contains at least one valid source ID and no invalid source IDs."""
        valid_ids = {result.get("source_id") for result in results if result.get("source_id")}
        cited_ids = re.findall(r"\[(S\d+)\]", answer)
        if not cited_ids:
            return "The generated answer contains no source citation."
        if any(cited_id not in valid_ids for cited_id in cited_ids):
            return "The generated answer cites a source ID that was not retrieved."
        return None

    @staticmethod
    def format_source_references(results: List[Dict[str, Any]]) -> str:
        """Format a clear list of source files and page numbers used to answer."""
        if not results:
            return ""
        seen_refs = []
        for r in results:
            sid = r.get("source_id", "S")
            fname = r.get("filename", "Document")
            page = r.get("page_number")
            page_str = f", page {page}" if page is not None else ""
            ref_str = f"[{sid}] {fname}{page_str}"
            if ref_str not in seen_refs:
                seen_refs.append(ref_str)
        return "\n\n**Sources:**\n" + "\n".join(f"- {ref}" for ref in seen_refs)

    @classmethod
    def cited_excerpt_fallback(cls, results: List[Dict[str, Any]]) -> Optional[str]:
        """Return a verbatim retrieved excerpt with its valid source citation."""
        for result in results:
            source_id = result.get("source_id")
            content = result.get("content")
            if not source_id or not content:
                continue
            excerpt = " ".join(str(content).split())
            if not excerpt:
                continue
            answer = (
                f"I couldn't generate a verified summary. The most relevant "
                f"retrieved excerpt from the document is: “{excerpt}” [{source_id}]."
            )
            if cls.citation_validation_error(answer, results) is None:
                return answer
        return None

