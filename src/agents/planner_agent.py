from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.contracts import PlannerOutput

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Intelligent query planning agent.

    Uses LLM structured reasoning to:
    1. Classify query intent ('standard', 'full_doc', 'section') dynamically from natural language.
    2. Extract section/chapter targets when the user asks to inspect a specific part.
    3. Reformulate follow-up queries and resolve conversational references into standalone search terms.
    """

    def __init__(self, llm_invoker: Optional[Callable[[list], str]] = None):
        self.llm_invoker = llm_invoker

    def reformulate_query(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
    ) -> str:
        """Resolve follow-up references into a standalone retrieval query using LLM."""
        if not self.llm_invoker or not chat_history:
            return query.strip()

        messages = [
            SystemMessage(
                content=(
                    "You are a search query reformulation specialist. "
                    "Rewrite the latest user question as one standalone search query for document retrieval. "
                    "Use prior turns to resolve references (such as 'what if it is smaller', 'in this case') "
                    "and ensure the specific regulation, rule number, or main subject discussed in earlier turns "
                    "is retained in the rewritten query. Return ONLY the search query without quotes or commentary."
                )
            )
        ]
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=human_msg))
            compact_ai = ai_msg[:120] + "..." if len(ai_msg) > 120 else ai_msg
            messages.append(AIMessage(content=compact_ai))
        messages.append(HumanMessage(content=query))

        try:
            rewritten = self.llm_invoker(messages)
            if rewritten and rewritten.strip():
                return rewritten.strip().strip('"')
        except Exception as e:
            logger.debug("Query reformulation error: %s", e)

        return query.strip()

    def plan_with_llm(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[PlannerOutput]:
        """Use LLM structured output to determine intent, target, and retrieval query."""
        if not self.llm_invoker:
            return None

        history = chat_history or []
        history_context = ""
        if history:
            turns = []
            for h, a in history[-3:]:
                compact_a = a[:150] + "..." if len(a) > 150 else a
                turns.append(f"User: {h}\nAssistant: {compact_a}")
            history_context = "\nRecent Conversation:\n" + "\n".join(turns) + "\n\n"

        prompt = (
            "You are an expert retrieval planner in a multi-agent document assistant.\n"
            "Analyze the latest user query and recent conversation history.\n\n"
            f"{history_context}"
            f"User Query: {query}\n\n"
            "Determine the following fields:\n"
            "1. 'intent':\n"
            "   - 'full_doc': User asks for a complete overview or summary of the entire document.\n"
            "   - 'section': User asks to summarize or inspect a specific chapter, section, table, or topic.\n"
            "   - 'standard': User asks a specific question about rules, data, facts, or instructions.\n"
            "2. 'target': If intent is 'section', specify the section/chapter name (e.g. 'chapter 4'); otherwise null.\n"
            "3. 'retrieval_query': Standalone search query optimized for semantic retrieval. "
            "Resolve any pronouns (it, that, these, former) into specific entities discussed earlier. "
            "Do not include conversational filler.\n\n"
            "Output ONLY a valid JSON object matching this exact structure:\n"
            '{"intent": "standard" | "full_doc" | "section", "target": string | null, "retrieval_query": "string"}'
        )

        try:
            raw_response = self.llm_invoker([SystemMessage(content=prompt)])
            if not raw_response or not isinstance(raw_response, str):
                return None

            clean_json = raw_response.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json)

            data = json.loads(clean_json)
            intent = data.get("intent", "standard")
            target = data.get("target")
            retrieval_query = str(data.get("retrieval_query") or query).strip()

            if intent == "full_doc":
                return PlannerOutput(
                    intent="full_doc",
                    target=None,
                    retrieval_query=retrieval_query,
                    start_page=1,
                    end_page=18,
                )
            elif intent == "section" and target:
                return PlannerOutput(
                    intent="section",
                    target=target,
                    retrieval_query=retrieval_query or f"Summary of {target}",
                )
            elif intent in ("standard", "full_doc", "section"):
                return PlannerOutput(
                    intent="standard",
                    target=None,
                    retrieval_query=retrieval_query,
                )
        except Exception as e:
            logger.debug("LLM planning fallback: %s", e)
            return None

        return None

    def plan(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> PlannerOutput:
        """
        Execute planning logic and produce a PlannerOutput.
        
        Routes single-turn queries cleanly without wasting LLM calls, and uses
        the LLM for multi-turn follow-up reference resolution when chat history exists.
        """
        history = chat_history or []
        lowered = query.casefold().strip()

        # 1. Multi-turn follow-ups: Use LLM to resolve references in conversation context
        if history and self.llm_invoker:
            llm_plan = self.plan_with_llm(query, history)
            if llm_plan is not None:
                return llm_plan

            recent_history = history[-2:]
            retrieval_query = self.reformulate_query(query, recent_history)
            return PlannerOutput(
                intent="standard",
                target=None,
                retrieval_query=retrieval_query,
            )

        # 2. Semantic intent detection across natural language variations
        section_match = re.search(
            r"\b(chapter\s*\w+|section\s*[\w\.]+|part\s*\w+|rule\s*[\w\.]+|table\s*\w+|appendix\s*\w+)\b",
            lowered,
        )

        is_summary_request = any(
            term in lowered
            for term in [
                "summary", "summarize", "overview", "gist", "brief",
                "takeaway", "takeaways", "recap", "synopsis", "outline",
                "breakdown", "highlights", "key points", "entire document",
                "whole document", "whole file", "all pages",
            ]
        )

        if is_summary_request:
            if section_match:
                target = section_match.group(1).strip()
                return PlannerOutput(
                    intent="section",
                    target=target,
                    retrieval_query=target,
                )
            return PlannerOutput(
                intent="full_doc",
                target=None,
                retrieval_query=query.strip(),
            )

        if section_match and any(w in lowered for w in ["show", "explain", "about", "details", "read", "view"]):
            target = section_match.group(1).strip()
            return PlannerOutput(
                intent="section",
                target=target,
                retrieval_query=target,
            )

        # 2. Multi-turn follow-ups: Use LLM to resolve references in conversation context
        if history and self.llm_invoker:
            llm_plan = self.plan_with_llm(query, history)
            if llm_plan is not None:
                return llm_plan

            recent_history = history[-2:]
            retrieval_query = self.reformulate_query(query, recent_history)
            return PlannerOutput(
                intent="standard",
                target=None,
                retrieval_query=retrieval_query,
            )

        # 3. Standard standalone query: optimal query is the query itself
        return PlannerOutput(
            intent="standard",
            target=None,
            retrieval_query=query.strip(),
        )

    # ── Backward-compatible helper aliases ──────────────────────────────────
    @staticmethod
    def expand_legal_exclusion_query(query: str) -> str:
        """Backward-compatible alias: returns clean query without hardcoded state entities."""
        return query.strip()

    @staticmethod
    def detect_summary_intent(query: str) -> Tuple[str, Optional[str]]:
        """Backward-compatible alias for summary intent detection."""
        lowered = query.casefold().strip()
        if "summary" in lowered or "summarize" in lowered or "overview" in lowered:
            match = re.search(r"\b(chapter\s*\d+|section\s*[\d\.]+|table\s*[0-9a-zA-Z]+)\b", lowered)
            if match:
                return "section", match.group(1).strip()
            if any(term in lowered for term in ["document", "pdf", "file", "overview", "entire", "whole"]):
                return "full_doc", None
        return "standard", None

    @staticmethod
    def needs_reformulation(query: str) -> bool:
        """Backward-compatible alias."""
        return True


# Backward-compatible alias
QueryPlanner = PlannerAgent
