from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.tools.table_tools import create_table_tools
from src.services.table_service import TableService

logger = logging.getLogger(__name__)


class TableAgent:
    """
    Deterministic numerical specialist equipped with exposed calculation tools.
    Performs exact cell lookups and Decimal/Fraction arithmetic without LLM approximation.
    """

    def __init__(
        self,
        table_service: Optional[TableService] = None,
        llm_invoker: Optional[Callable[[list], str]] = None,
    ):
        self.table_service = table_service or TableService()
        self.llm_invoker = llm_invoker
        # Expose dedicated table tools directly to this agent
        self.tools = create_table_tools(self.table_service)
        self._tool_map = {t.name: t for t in self.tools}

    def has_table_intent(self, query: str, evidence_chunks: List[Dict[str, Any]]) -> bool:
        """Check if query is asking for tabular math or cell extraction."""
        has_table_chunks = any(c.get("content_type") == "table" for c in evidence_chunks)
        if not has_table_chunks and not evidence_chunks:
            return False
        return self.table_service.has_table_aggregation_keyword(query) or has_table_chunks

    def analyze_and_calculate(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Evaluate tabular data and execute calculation or lookup using exposed tools.
        """
        if not evidence_chunks:
            return None

        has_table_results = any(r.get("content_type") == "table" for r in evidence_chunks)

        if has_table_results and self.table_service.has_table_aggregation_keyword(query):
            if self.table_service.has_explicit_table_aggregation_intent(query, evidence_chunks):
                calc_tool = self._tool_map.get("calculate_table_aggregation")
                if calc_tool:
                    return calc_tool.invoke({
                        "query": query,
                        "table_chunks": evidence_chunks,
                    })
            else:
                lookup_tool = self._tool_map.get("lookup_table_cell")
                if lookup_tool:
                    return lookup_tool.invoke({
                        "query": query,
                        "table_chunks": evidence_chunks,
                    })
        else:
            lookup_tool = self._tool_map.get("lookup_table_cell")
            if lookup_tool:
                return lookup_tool.invoke({
                    "query": query,
                    "table_chunks": evidence_chunks,
                })

        return None

