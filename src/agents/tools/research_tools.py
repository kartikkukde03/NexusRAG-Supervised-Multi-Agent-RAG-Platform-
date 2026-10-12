"""
src/agents/tools/research_tools.py

Dedicated @tool definitions exposed exclusively to ResearchAgent.

Tools (per architecture_plan.md):
  - search_documents    — hybrid vector + BM25 retrieval
  - get_page_range      — retrieve chunks from a page range
  - get_section         — retrieve chunks by section/chapter keyword
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool


def create_research_tools(vectorstore: Any) -> List[Any]:
    """
    Creates and returns standard LangChain @tool definitions exposed
    specifically to the ResearchAgent for document exploration.
    """

    @tool
    def search_documents(
        query: str,
        document_scope: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search document text using hybrid vector and keyword retrieval.

        Args:
            query: The search phrase or question.
            document_scope: Optional filename to restrict search to a specific document.
            top_k: Number of chunks to retrieve.
        """
        if document_scope:
            return vectorstore.query(
                query_text=query,
                filename=document_scope,
                top_k=top_k,
            )
        return vectorstore.query_all(
            query_text=query,
            top_k=top_k,
        )

    @tool
    def get_page_range(
        filename: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_chunks: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific page range of a document.

        Args:
            filename: Target document name.
            start_page: Starting page number (inclusive).
            end_page: Ending page number (inclusive). Defaults to document page count.
            max_chunks: Maximum chunks to return.
        """
        end_p = end_page
        if end_p is None:
            if hasattr(vectorstore, "get_document_outline"):
                try:
                    end_p = vectorstore.get_document_outline(filename).get("page_count", 500)
                except Exception:
                    end_p = 500
            else:
                end_p = 500
        return vectorstore.query_by_pages(
            filename=filename,
            start_page=start_page,
            end_page=end_p,
            max_chunks=max_chunks,
        )

    @tool
    def get_section(
        filename: str,
        section_keyword: str,
        max_chunks: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks belonging to a specific chapter, section, or topic heading.

        Args:
            filename: Target document name.
            section_keyword: Section title or number (e.g., 'Chapter 3', 'Rule 4.2').
            max_chunks: Maximum chunks to return.
        """
        return vectorstore.query_by_section(
            filename=filename,
            section_keyword=section_keyword,
            max_chunks=max_chunks,
        )

    @tool
    def get_document_outline(
        filename: str,
    ) -> Dict[str, Any]:
        """
        Retrieve structural outline, page count, detected sections, and multi-page milestones across the entire document.

        Args:
            filename: Target document name.
        """
        if hasattr(vectorstore, "get_document_outline"):
            return vectorstore.get_document_outline(filename=filename)
        return {"filename": filename, "total_chunks": 0, "page_count": 0, "sections": [], "outline_chunks": []}

    return [search_documents, get_page_range, get_section, get_document_outline]
