from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MAX_HISTORY_TURNS = 10
MAX_CONTEXT_CHARS = 30000


class WriterAgent:
    """Specialist agent responsible for composing grounded answers from retrieved evidence."""

    def __init__(
        self,
        llm_invoker: Optional[Callable[[list], str]] = None,
        llm_streamer: Optional[Callable[[list], Generator[str, None, None]]] = None,
    ):
        self.llm_invoker = llm_invoker
        self.llm_streamer = llm_streamer

    @staticmethod
    def build_context_and_evidence(
        results: List[Dict[str, Any]],
        default_filename: Optional[str] = None,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Construct context text and ensure immutable source IDs."""
        context_parts = []
        context_results = []
        remaining_chars = max_context_chars

        for result in results:
            if not result.get("content"):
                continue
            if remaining_chars <= 0:
                break
            result_copy = dict(result)
            result_copy["source_id"] = f"S{len(context_results) + 1}"
            filename = result_copy.get("filename") or default_filename or "Unknown"
            page_number = result_copy.get("page_number")
            page_label = f", page {page_number}" if page_number is not None else ""
            source_text = (
                f"[{result_copy['source_id']}: {filename}{page_label}]\n"
                f"{result_copy['content']}"
            )
            source_text = source_text[:remaining_chars]
            remaining_chars -= len(source_text)
            context_parts.append(source_text)
            context_results.append(result_copy)

        return "\n\n---\n\n".join(context_parts), context_results

    @staticmethod
    def build_system_prompt(
        context: str,
        results: List[Dict[str, Any]],
        document_scope: Optional[str] = None,
    ) -> str:
        scope = document_scope or "all stored documents"
        source_ids_available = ", ".join(
            r.get("source_id", "S?") for r in results if r.get("source_id")
        ) or "none"

        return (
            "You are an expert, thorough document assistant. Provide a COMPREHENSIVE, "
            "DETAILED, and WELL-EXPLAINED long-form answer grounded strictly in the supplied "
            "evidence below. Do NOT give brief 1-2 sentence answers.\n\n"
            "## ANSWER STYLE & STRUCTURE\n"
            "- Explain the core rule, definition, or answer thoroughly.\n"
            "- Break down the explanation using structured headings, bullet points, and numbered lists where appropriate.\n"
            "- Include all relevant conditions, exceptions, notes, numerical thresholds, and context from the document.\n"
            "- If tables are present, clearly explain what each column and row signifies.\n"
            "- Explain *why* and *how* the regulation or concept applies based on the text.\n\n"
            "## NUMERICAL REASONING & RANGE BRACKETS\n"
            "- If the question specifies a particular numerical value or threshold:\n"
            "  * Match the user's value against the relevant bracket, row, tier, or category in the text or tables.\n"
            "  * State the exact rule or formula for that bracket, and calculate required values or percentages step-by-step.\n"
            "  * Never state that the document lacks information if the user's specific scenario falls within a defined bracket or rule.\n\n"
            "## CITATION RULES (MANDATORY)\n"
            "- Cite every factual claim, sentence, or bullet with its exact source ID, e.g., [S1], [S2].\n"
            f"- Available source IDs: {source_ids_available}\n"
            "- Do NOT invent source IDs. Only use the ones listed above.\n\n"
            "## INSUFFICIENT EVIDENCE RULE\n"
            "- ONLY if the retrieved evidence genuinely has NO relevance, rule, or bracket pertaining to the topic, "
            "state: 'The provided document does not contain information regarding [user's current question topic].'\n\n"
            f"Search scope: '{scope}'\n\nContext:\n{context}"
        )

    @staticmethod
    def build_summary_system_prompt(
        context: str,
        results: List[Dict[str, Any]],
        document_scope: Optional[str] = None,
    ) -> str:
        scope = document_scope or "the uploaded document"
        source_ids_available = ", ".join(
            r.get("source_id", "S?") for r in results if r.get("source_id")
        ) or "none"

        return (
            "You are an expert executive document analyst. Produce a COMPREHENSIVE, STRUCTURED, "
            "and HIGH-VALUE WHOLE-DOCUMENT SUMMARY grounded strictly in the multi-page evidence below.\n\n"
            "## REQUIRED SUMMARY STRUCTURE\n"
            "1. 📋 **Executive Overview & Scope**: Purpose of the document, governing authority/organization, and main objective.\n"
            "2. 📑 **Structural Breakdown & Major Sections**: Break down the core chapters, sections, or parts and what each covers.\n"
            "3. ⚖️ **Core Regulations, Findings & Metrics**: The most significant rules, quantitative thresholds, data points, or formulas.\n"
            "4. 🚫 **Applicability, Exceptions & Exclusions**: What is explicitly included or exempted from the scope.\n"
            "5. 🎯 **Key Directives & Conclusion**: High-level conclusions or next steps.\n\n"
            "## CITATION RULES (MANDATORY)\n"
            "- Cite every bullet point and major claim with its exact source ID, e.g., [S1], [S2].\n"
            f"- Available source IDs: {source_ids_available}\n"
            "- Do NOT invent source IDs. Only use the ones listed above.\n\n"
            f"Document scope: '{scope}'\n\nContext:\n{context}"
        )

    @classmethod
    def build_messages(
        cls,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]],
        results: List[Dict[str, Any]],
        context: str,
        document_scope: Optional[str] = None,
        is_full_summary: bool = False,
    ) -> List[Any]:
        history = chat_history[-MAX_HISTORY_TURNS:] if chat_history else []
        if is_full_summary:
            system_prompt = cls.build_summary_system_prompt(context, results, document_scope)
        else:
            system_prompt = cls.build_system_prompt(context, results, document_scope)
        messages = [SystemMessage(content=system_prompt)]

        if history:
            for human_msg, ai_msg in history[-5:]:
                messages.append(HumanMessage(content=human_msg))
                concise_ai = ai_msg if len(ai_msg) <= 350 else ai_msg[:350] + "..."
                messages.append(AIMessage(content=concise_ai))
        messages.append(HumanMessage(content=query))
        return messages

    def write_answer(self, messages: List[Any]) -> str:
        """Invoke LLM to generate candidate answer."""
        if not self.llm_invoker:
            raise ValueError("No LLM invoker configured for WriterAgent.")
        return self.llm_invoker(messages)

    def write_answer_stream(self, messages: List[Any]) -> Generator[str, None, None]:
        """Stream tokens directly from LLM via llm_streamer or yield full response."""
        if self.llm_streamer:
            yield from self.llm_streamer(messages)
        elif self.llm_invoker:
            yield self.llm_invoker(messages)
        else:
            raise ValueError("No LLM invoker or streamer configured for WriterAgent.")


# Backward-compatible alias
AnswerWriterAgent = WriterAgent

