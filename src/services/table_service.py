from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple


class TableService:
    """Deterministic service for exact table cell lookups and decimal-safe aggregations."""

    @staticmethod
    def normalize_table_text(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9%]+", value.casefold()))

    @staticmethod
    def is_numeric_table_value(value: str) -> bool:
        normalized = value.strip().replace(",", "")
        return bool(
            re.fullmatch(
                r"(?:[+-]?\d+(?:\.\d+)?%?|\(\d+(?:\.\d+)?\))",
                normalized,
            )
        )

    @staticmethod
    def aggregation_operation(query: str) -> Optional[str]:
        operation_patterns = {
            "sum": r"\b(?:sum|total)\b",
            "average": r"\b(?:average|mean)\b",
            "minimum": r"\b(?:minimum|min)\b",
            "maximum": r"\b(?:maximum|max)\b",
            "count": r"\bcount\b",
        }
        operations = [
            operation
            for operation, pattern in operation_patterns.items()
            if re.search(pattern, query, flags=re.IGNORECASE)
        ]
        return operations[0] if len(operations) == 1 else None

    @classmethod
    def has_table_aggregation_keyword(cls, query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:sum|total|average|mean|minimum|maximum|min|max|count)\b",
                query,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def has_explicit_table_aggregation_intent(
        cls,
        query: str,
        results: List[Dict[str, Any]],
    ) -> bool:
        operation = cls.aggregation_operation(query)
        if operation is None:
            return False
        operation_terms = {
            "sum": {"sum", "total"},
            "average": {"average", "mean"},
            "minimum": {"minimum", "min"},
            "maximum": {"maximum", "max"},
            "count": {"count"},
        }[operation]
        query_tokens = cls.normalize_table_text(query).split()
        operation_positions = [
            index for index, token in enumerate(query_tokens) if token in operation_terms
        ]
        for result in results:
            if result.get("content_type") != "table":
                continue
            header_line = next(
                (
                    line
                    for line in result.get("table_header_markdown", "").splitlines()
                    if line.strip().startswith("|")
                    and not all(char in "|- :" for char in line.strip())
                ),
                "",
            )
            headers = [
                cell.strip()
                for cell in header_line.strip().strip("|").split("|")
            ]
            header_spans = []
            for header in headers:
                header_tokens = cls.normalize_table_text(header).split()
                if not header_tokens:
                    continue
                header_spans.extend(
                    range(index, index + len(header_tokens))
                    for index in range(len(query_tokens) - len(header_tokens) + 1)
                    if query_tokens[index : index + len(header_tokens)] == header_tokens
                )
            header_positions = {
                position for span in header_spans for position in span
            }
            if any(position not in header_positions for position in operation_positions):
                return True
        return False

    @staticmethod
    def table_decimal(value: str) -> Decimal:
        normalized = value.strip().replace(",", "").rstrip("%")
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = f"-{normalized[1:-1]}"
        try:
            return Decimal(normalized)
        except InvalidOperation as exc:
            raise ValueError(f"Unsupported numeric table value: {value}") from exc

    @staticmethod
    def format_decimal(value: Decimal) -> str:
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return format(normalized, ".0f")
        return format(normalized, "f").rstrip("0").rstrip(".")

    @classmethod
    def format_fraction(cls, value: Fraction) -> str:
        denominator = value.denominator
        twos = 0
        fives = 0
        while denominator % 2 == 0:
            denominator //= 2
            twos += 1
        while denominator % 5 == 0:
            denominator //= 5
            fives += 1
        if denominator != 1:
            return f"{value.numerator}/{value.denominator}"
        scale = max(twos, fives)
        scaled_numerator = value.numerator * 2 ** (scale - twos) * 5 ** (scale - fives)
        return cls.format_decimal(Decimal(scaled_numerator).scaleb(-scale))

    def try_table_aggregation(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Calculate only explicit row selections wholly present in one table chunk."""
        operation = self.aggregation_operation(query)
        if operation is None:
            return None
        normalized_query = self.normalize_table_text(query)
        query_tokens = set(normalized_query.split())
        stop_words = {
            "a", "across", "all", "and", "calculate", "compute", "count",
            "data", "each", "find", "for", "from", "give", "in", "me",
            "of", "please", "row", "rows", "show", "the", "their", "to",
            "value", "values", "what", "which", "with",
        }
        query_tokens -= {
            "sum", "total", "average", "mean", "minimum", "maximum",
            "min", "max", "count",
        }

        candidates = []
        for result in results:
            if result.get("content_type") != "table":
                continue
            header_line = next(
                (
                    line
                    for line in result.get("table_header_markdown", "").splitlines()
                    if line.strip().startswith("|")
                    and not all(char in "|- :" for char in line.strip())
                ),
                "",
            )
            headers = [
                cell.strip()
                for cell in header_line.strip().strip("|").split("|")
            ]
            if len(headers) < 2:
                continue
            matching_columns = [
                index
                for index, header in enumerate(headers)
                if self.normalize_table_text(header)
                and f" {self.normalize_table_text(header)} " in f" {normalized_query} "
            ]
            if len(matching_columns) != 1:
                continue
            column_index = matching_columns[0]
            normalized_headers = self.normalize_table_text(headers[column_index]).split()

            table_rows = []
            for line in result.get("content", "").splitlines():
                stripped_line = line.strip()
                if not stripped_line.startswith("|"):
                    continue
                cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
                if len(cells) != len(headers):
                    continue
                if all(set(cell) <= {"-", " "} for cell in cells):
                    continue
                if [self.normalize_table_text(cell) for cell in cells] == [
                    self.normalize_table_text(header) for header in headers
                ]:
                    continue
                row_label = cells[0]
                normalized_label = self.normalize_table_text(row_label)
                if normalized_label and f" {normalized_label} " in f" {normalized_query} ":
                    table_rows.append((row_label, normalized_label, cells[column_index]))

            if not table_rows:
                continue
            normalized_labels = [label for _, label, _ in table_rows]
            if len(normalized_labels) != len(set(normalized_labels)):
                continue
            if any(
                left != right and (f" {left} " in f" {right} " or f" {right} " in f" {left} ")
                for index, left in enumerate(normalized_labels)
                for right in normalized_labels[index + 1 :]
            ):
                continue

            selected_label_tokens = {
                token for _, label, _ in table_rows for token in label.split()
            }
            remaining_tokens = (
                query_tokens
                - set(normalized_headers)
                - selected_label_tokens
                - stop_words
            )
            if remaining_tokens:
                continue

            try:
                selected_values = [
                    (label, raw_value, self.table_decimal(raw_value))
                    for label, _, raw_value in table_rows
                    if self.is_numeric_table_value(raw_value)
                ]
            except ValueError:
                continue
            if len(selected_values) != len(table_rows):
                continue
            candidates.append((result, headers[column_index], selected_values))

        if len(candidates) != 1:
            return None

        result, column_label, selected_values = candidates[0]
        source_id = result.get("source_id")
        if not source_id:
            return None
        inputs = " and ".join(
            f"{row_label} ({raw_value})"
            for row_label, raw_value, _ in selected_values
        )
        raw_values_are_percentages = [
            raw_value.strip().endswith("%")
            for _, raw_value, _ in selected_values
        ]
        if any(raw_values_are_percentages) and not all(raw_values_are_percentages):
            return None
        suffix = "%" if all(raw_values_are_percentages) else ""
        values = [value for _, _, value in selected_values]

        if operation == "sum":
            answer = self.format_decimal(sum(values, Decimal(0))) + suffix
            return f"Sum of {column_label} for {inputs} = {answer} [{source_id}]."
        if operation == "average":
            average = Fraction(sum(values, Decimal(0))) / len(values)
            answer = self.format_fraction(average) + suffix
            return f"Average of {column_label} for {inputs} = {answer} [{source_id}]."
        if operation == "count":
            return (
                f"Count of {column_label} values for {inputs} = "
                f"{len(values)} [{source_id}]."
            )

        target = min(values) if operation == "minimum" else max(values)
        matching_rows = [
            f"{row_label} ({raw_value})"
            for row_label, raw_value, value in selected_values
            if value == target
        ]
        row_answers = " and ".join(matching_rows)
        label = "Minimum" if operation == "minimum" else "Maximum"
        return (
            f"{label} of {column_label} among {inputs} is "
            f"{row_answers} [{source_id}]."
        )

    def try_table_cell_lookup(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Answer only an unambiguous exact row/column lookup from retrieved tables."""
        normalized_query = f" {self.normalize_table_text(query)} "
        matches = []
        for result in results:
            if result.get("content_type") != "table":
                continue

            headers_markdown = result.get("table_header_markdown", "")
            header_line = next(
                (
                    line
                    for line in headers_markdown.splitlines()
                    if line.strip().startswith("|")
                ),
                "",
            )
            headers = [
                cell.strip()
                for cell in header_line.strip().strip("|").split("|")
            ]
            if len(headers) < 2:
                continue
            matching_columns = [
                index
                for index, header in enumerate(headers)
                if self.normalize_table_text(header)
                and f" {self.normalize_table_text(header)} " in normalized_query
            ]
            if len(matching_columns) != 1:
                continue

            for line in result.get("content", "").splitlines():
                stripped_line = line.strip()
                if not stripped_line.startswith("|"):
                    continue
                cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
                if len(cells) != len(headers) or all(
                    set(cell) <= {"-", " "} for cell in cells
                ):
                    continue
                row_label = cells[0]
                normalized_label = self.normalize_table_text(row_label)
                if not normalized_label or f" {normalized_label} " not in normalized_query:
                    continue
                value = cells[matching_columns[0]]
                if self.is_numeric_table_value(value):
                    matches.append(
                        (
                            result,
                            row_label,
                            headers[matching_columns[0]],
                            value,
                        )
                    )

        if len(matches) != 1:
            return None

        result, row_label, column_label, value = matches[0]
        return (
            f"{row_label} — {column_label}: {value} "
            f"[{result['source_id']}]."
        )

