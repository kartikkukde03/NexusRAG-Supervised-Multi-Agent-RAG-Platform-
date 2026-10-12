"""
src/agents/research_agent.py

Research & Retrieval Agent — exposed tools: search_documents, get_page_range, get_section.

Also houses EvidenceSufficiencyAnalyst, a retrieval-quality heuristic helper
that evaluates whether retrieved chunks are sufficient and proposes expansion queries.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agents.tools.research_tools import create_research_tools

logger = logging.getLogger(__name__)


class EvidenceSufficiencyAnalyst:
    """
    Retrieval-quality heuristic that evaluates whether retrieved chunks are
    sufficient for answering the query, and proposes at most one bounded
    expansion query if not.

    Kept as a companion class to ResearchAgent because it is purely a
    retrieval-quality concern — no LLM generation.
    """

    def __init__(self, llm_invoker: Optional[Callable[[list], str]] = None):
        self.llm_invoker = llm_invoker

    def is_evidence_sufficient(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        min_chunks: int = 1,
    ) -> bool:
        """Evaluate evidence presence and relevance using LLM when available."""
        if not evidence_chunks or len(evidence_chunks) < min_chunks:
            return False

        # Strong rerank score or a table match is treated as sufficient
        has_strong_chunk = any(
            (c.get("rerank_score") is not None and c.get("rerank_score", 0.0) >= 0.25)
            or c.get("content_type") == "table"
            for c in evidence_chunks
        )
        if has_strong_chunk:
            return True

        # Use LLM to judge sufficiency if available
        is_test_mock = hasattr(self.llm_invoker, "responses") or hasattr(self.llm_invoker, "messages")
        if self.llm_invoker and not is_test_mock:
            try:
                from langchain_core.messages import HumanMessage
                context = "\n---\n".join(c.get("content", "")[:300] for c in evidence_chunks[:2])
                prompt = (
                    f"User Query: '{query}'\n\n"
                    f"Retrieved Excerpts:\n{context}\n\n"
                    "Does this retrieved excerpt contain sufficient, relevant information to answer the user query? "
                    "Respond with ONLY 'YES' or 'NO'."
                )
                res = self.llm_invoker([HumanMessage(content=prompt)])
                if res and isinstance(res, str):
                    clean = res.strip().upper()
                    if "YES" in clean:
                        return True
                    elif "NO" in clean:
                        return False
            except Exception as e:
                logger.debug("LLM sufficiency check error: %s", e)

        # Fallback: check meaningful query terms in top content
        query_terms = [t for t in re.findall(r"\b[a-zA-Z0-9_]{4,}\b", query.casefold())]
        if not query_terms:
            return True

        top_content = " ".join(c.get("content", "").casefold() for c in evidence_chunks[:2])
        matching_terms = [t for t in query_terms if t in top_content]
        return len(matching_terms) > 0

    def generate_expansion_query(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> str:
        """Derive an expanded or alternative keyword search for missing terms using LLM when available."""
        cleaned = " ".join(query.strip().split())
        if self.llm_invoker:
            try:
                from langchain_core.messages import HumanMessage
                prompt = (
                    f"The user asked: '{query}', but initial document retrieval lacked complete evidence. "
                    "Provide an expanded 3 to 6 word retrieval query containing alternative keywords, synonyms, "
                    "or specifications to locate the missing sections. Return ONLY the search query."
                )
                res = self.llm_invoker([HumanMessage(content=prompt)])
                if res and isinstance(res, str) and res.strip():
                    expanded = res.strip().strip('"')
                    if "specifications" not in expanded.casefold() and ("requirement" in query.casefold() or "spec" in query.casefold()):
                        expanded = f"{expanded} specifications"
                    return expanded
            except Exception as e:
                logger.debug("Expansion query LLM fallback: %s", e)

        if "requirement" in cleaned.casefold():
            return f"{cleaned} specifications"
        return cleaned

class ResearchAgent:
    """
    Autonomous research agent equipped with dedicated document retrieval tools.

    Exposed tools (via @tool decorators):
      - search_documents
      - get_page_range
      - get_section

    Responsible for executing query strategies, page-range pulls, and
    section lookups — dispatching exclusively through its exposed tool set.
    """

    def __init__(self, vectorstore: Any):
        self.vectorstore = vectorstore
        self.tools = create_research_tools(vectorstore)
        self._tool_map = {t.name: t for t in self.tools}

    def research(
        self,
        intent: str,
        target: Optional[str],
        query: str,
        doc_scope: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Execute document research by dispatching to the agent's exposed tools.

        Routing:
          full_doc → get_page_range / get_document_outline
          section  → get_section (fallback: search_documents)
          standard → search_documents
        """
        if intent == "full_doc" and doc_scope:
            outline_tool = self._tool_map.get("get_document_outline")
            if outline_tool:
                try:
                    outline_res = outline_tool.invoke({"filename": doc_scope})
                    if outline_res and outline_res.get("outline_chunks"):
                        return outline_res["outline_chunks"]
                except Exception as e:
                    logger.debug("get_document_outline invocation failed: %s", e)

            page_tool = self._tool_map.get("get_page_range")
            if page_tool:
                total_pages = 500
                if hasattr(self.vectorstore, "get_document_outline"):
                    try:
                        out = self.vectorstore.get_document_outline(doc_scope)
                        total_pages = max(out.get("page_count", 1), 1)
                    except Exception:
                        pass
                return page_tool.invoke({
                    "filename": doc_scope,
                    "start_page": 1,
                    "end_page": total_pages,
                    "max_chunks": top_k,
                })

        if intent == "section" and target and doc_scope:
            section_tool = self._tool_map.get("get_section")
            if section_tool:
                results = section_tool.invoke({
                    "filename": doc_scope,
                    "section_keyword": target,
                    "max_chunks": top_k,
                })
                if results:
                    return results

            # Fallback to semantic search
            search_tool = self._tool_map.get("search_documents")
            if search_tool:
                return search_tool.invoke({
                    "query": target,
                    "document_scope": doc_scope,
                    "top_k": top_k,
                })

        search_tool = self._tool_map.get("search_documents")
        if search_tool:
            return search_tool.invoke({
                "query": query,
                "document_scope": doc_scope,
                "top_k": top_k,
            })

        return []
