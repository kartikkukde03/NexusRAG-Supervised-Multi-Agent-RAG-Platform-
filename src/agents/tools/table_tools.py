from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from src.services.table_service import TableService


def create_table_tools(table_service: Optional[TableService] = None) -> List[Any]:
    """
    Creates and returns standard LangChain @tool definitions exposed
    specifically to the TableAgent for deterministic math calculations.
    """
    tbl = table_service or TableService()

    @tool
    def calculate_table_aggregation(
        query: str,
        table_chunks: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Compute exact Decimal and Fraction arithmetic operations (sum, average, min, max, count)
        on numerical columns across Markdown tables.

        Args:
            query: The user query asking for an aggregation (e.g. 'sum of population').
            table_chunks: List of retrieved table chunks.
        """
        if not table_chunks:
            return None
        return tbl.try_table_aggregation(query, table_chunks)

    @tool
    def lookup_table_cell(
        query: str,
        table_chunks: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Extract specific cell values from a Markdown table matching a row or entity in the query.

        Args:
            query: Query referencing a specific row or category.
            table_chunks: List of retrieved table chunks.
        """
        if not table_chunks:
            return None
        return tbl.try_table_cell_lookup(query, table_chunks)

    return [calculate_table_aggregation, lookup_table_cell]

