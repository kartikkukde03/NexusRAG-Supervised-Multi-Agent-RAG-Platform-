from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import StreamWriter

from src.agents.citation_agent import CitationAgent
from src.agents.contracts import AgentQueryResult, WorkflowState
from src.agents.research_agent import EvidenceSufficiencyAnalyst
from src.agents.planner_agent import PlannerAgent
from src.agents.research_agent import ResearchAgent
from src.agents.summary_agent import DocumentSummaryAgent
from src.agents.table_agent import TableAgent
from src.agents.writer_agent import WriterAgent
from src.services.citation_service import CitationService
from src.services.table_service import TableService

logger = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = "The provided excerpt does not contain information regarding"
CITATION_FAILURE_MESSAGE = (
    "I couldn't provide a verified answer because the response did not cite "
    "valid retrieved sources. Please try rephrasing the question."
)


class MultiAgentRAGGraph:
    """
    Supervised LangGraph workflow orchestrating specialized agents:
    - PlannerAgent: Query decomposition and intent routing.
    - ResearchAgent: Autonomous document retrieval via exposed research tools.
    - TableAgent: Tabular arithmetic and cell lookups via exposed table tools.
    - WriterAgent: Grounded answer drafting adhering to source citations.
    - CitationAgent: Verification and self-healing repair via exposed citation tools.
    - EvidenceSufficiencyAnalyst: Quality audit and bounded secondary expansion.
    """

    def __init__(
        self,
        vectorstore: Any,
        llm_invoker: Callable[[list], str],
        table_service: Optional[TableService] = None,
        citation_service: Optional[CitationService] = None,
        enable_evidence_analyst: bool = False,
        llm_streamer: Optional[Callable[[list], Any]] = None,
    ):
        self.vectorstore = vectorstore
        self.llm_invoker = llm_invoker
        self.table_service = table_service or TableService()
        self.citation_service = citation_service or CitationService()
        self.enable_evidence_analyst = enable_evidence_analyst
        self._last_result: Optional[AgentQueryResult] = None

        # Build streaming wrapper if not provided
        if llm_streamer is None:
            if hasattr(llm_invoker, "stream"):
                def _stream_from_obj(msgs):
                    for chunk in llm_invoker.stream(msgs):
                        if hasattr(chunk, "content"):
                            yield chunk.content
                        elif hasattr(chunk, "text"):
                            yield chunk.text
                        else:
                            yield str(chunk)
                llm_streamer = _stream_from_obj
            elif callable(llm_invoker):
                def _stream_from_inv(msgs):
                    yield llm_invoker(msgs)
                llm_streamer = _stream_from_inv

        self.llm_streamer = llm_streamer

        # Instantiate decoupled specialized agents with their dedicated exposed tools
        self.planner = PlannerAgent(llm_invoker=self.llm_invoker)
        self.research_agent = ResearchAgent(vectorstore=self.vectorstore)
        self.table_agent = TableAgent(
            table_service=self.table_service,
            llm_invoker=self.llm_invoker,
        )
        self.writer_agent = WriterAgent(
            llm_invoker=self.llm_invoker,
            llm_streamer=self.llm_streamer,
        )
        self.citation_agent = CitationAgent(
            llm_invoker=self.llm_invoker,
            citation_service=self.citation_service,
        )
        self.evidence_analyst = EvidenceSufficiencyAnalyst(llm_invoker=self.llm_invoker)
        self.summary_agent = DocumentSummaryAgent(
            vectorstore=self.vectorstore,
            llm_invoker=self.llm_invoker,
        )

        # Backward compatibility aliases for existing helper calls
        self.answer_writer = self.writer_agent
        self.citation_repair = self.citation_agent

        # LangGraph in-memory checkpointer for stateful session persistence
        self.checkpointer = MemorySaver()

        self.graph = self._build_graph()

    def _planner_node(self, state: WorkflowState) -> WorkflowState:
        query = state["query"]
        history = state.get("chat_history", [])
        plan_output = self.planner.plan(query, history)

        trace = list(state.get("trace_steps", []))
        trace.append("planner")
        details = dict(state.get("trace_details", {}))
        details["planner"] = {
            "intent": plan_output.intent,
            "target": plan_output.target,
            "retrieval_query": plan_output.retrieval_query,
        }

        return {
            "planner_intent": plan_output.intent,
            "planner_target": plan_output.target,
            "retrieval_query": plan_output.retrieval_query,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _execute_retrieval(
        self,
        intent: str,
        target: Optional[str],
        query_text: str,
        doc_scope: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Dispatch retrieval to the DocumentSummaryAgent or ResearchAgent via exposed tools."""
        if intent == "full_doc":
            resolved_scope = doc_scope
            if not resolved_scope and hasattr(self.vectorstore, "list_files"):
                available = self.vectorstore.list_files()
                for f in available:
                    if f.lower() in query_text.lower():
                        resolved_scope = f
                        break
                if not resolved_scope and len(available) == 1:
                    resolved_scope = available[0]

            if resolved_scope:
                return self.summary_agent.gather_full_document_evidence(
                    filename=resolved_scope,
                    query=query_text,
                    max_total_chunks=max(top_k, 18),
                )
        return self.research_agent.research(
            intent=intent,
            target=target,
            query=query_text,
            doc_scope=doc_scope,
            top_k=top_k,
        )

    def _retrieval_node(self, state: WorkflowState) -> WorkflowState:
        intent = state.get("planner_intent", "standard")
        target = state.get("planner_target")
        query_text = state.get("retrieval_query") or state["query"]
        doc_scope = state.get("document_scope")
        top_k = state.get("top_k", 10)

        # ResearchAgent explores documents via its exposed research tools
        results = self._execute_retrieval(intent, target, query_text, doc_scope, top_k)

        context_text, evidence_chunks = self.writer_agent.build_context_and_evidence(
            results, default_filename=doc_scope
        )

        # TableAgent evaluates if tabular math tools should run
        table_answer = self.table_agent.analyze_and_calculate(state["query"], evidence_chunks)

        trace = list(state.get("trace_steps", []))
        trace.append("retrieval")
        details = dict(state.get("trace_details", {}))
        details["retrieval"] = {
            "chunks_retrieved": len(evidence_chunks),
            "scope": doc_scope or "all documents",
            "method": "Hierarchical Map-Reduce (100% chunks scanned & distilled across all sections)" if intent == "full_doc" else "Semantic Hybrid Search",
            "has_table": any(r.get("content_type") == "table" for r in evidence_chunks),
            "tools_used": [t.name for t in self.research_agent.tools],
        }

        return {
            "evidence_chunks": evidence_chunks,
            "context_text": context_text,
            "table_answer": table_answer,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _route_after_retrieval(self, state: WorkflowState) -> str:
        evidence = state.get("evidence_chunks", [])
        if not evidence:
            return "no_evidence"

        # Check for bounded evidence expansion if enabled
        if self.enable_evidence_analyst and state.get("expansion_count", 0) < 1:
            if not self.evidence_analyst.is_evidence_sufficient(state["query"], evidence):
                return "evidence_analyst"

        if state.get("table_answer"):
            return "table"
        return "answer_writer"

    def _evidence_analyst_node(self, state: WorkflowState) -> WorkflowState:
        query = state["query"]
        doc_scope = state.get("document_scope")
        top_k = state.get("top_k", 10)
        existing_evidence = state.get("evidence_chunks", [])

        expansion_query = self.evidence_analyst.generate_expansion_query(query, existing_evidence)
        new_results = self._execute_retrieval("standard", None, expansion_query, doc_scope, top_k)

        # Merge new results with existing
        seen_contents = {c.get("content", "").strip() for c in existing_evidence}
        combined = list(existing_evidence)
        for r in new_results:
            content = r.get("content", "").strip()
            if content and content not in seen_contents:
                seen_contents.add(content)
                combined.append(r)

        context_text, merged_evidence = self.writer_agent.build_context_and_evidence(
            combined, default_filename=doc_scope
        )

        trace = list(state.get("trace_steps", []))
        trace.append("evidence_analyst")
        details = dict(state.get("trace_details", {}))
        details["evidence_analyst"] = {
            "expansion_query": expansion_query,
            "expanded_chunks": len(merged_evidence),
        }

        return {
            "evidence_chunks": merged_evidence,
            "context_text": context_text,
            "expansion_count": state.get("expansion_count", 0) + 1,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _route_after_evidence_analyst(self, state: WorkflowState) -> str:
        if state.get("table_answer"):
            return "table"
        return "answer_writer"

    def _table_node(self, state: WorkflowState, writer: StreamWriter) -> WorkflowState:
        table_ans = state.get("table_answer", "")
        if table_ans:
            writer(table_ans)
        trace = list(state.get("trace_steps", []))
        trace.append("table_analyst")
        details = dict(state.get("trace_details", {}))
        details["table_analyst"] = {
            "output": table_ans,
            "tools_used": [t.name for t in self.table_agent.tools],
        }

        return {
            "verified_answer": table_ans,
            "citation_error": None,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _no_evidence_node(self, state: WorkflowState, writer: StreamWriter) -> WorkflowState:
        query = state["query"]
        ans = f"{NO_EVIDENCE_MESSAGE} {query.strip()}."
        writer(ans)
        trace = list(state.get("trace_steps", []))
        trace.append("no_evidence")
        details = dict(state.get("trace_details", {}))
        details["outcome"] = "no_evidence"

        return {
            "verified_answer": ans,
            "citation_error": None,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _answer_writer_node(self, state: WorkflowState, writer: StreamWriter) -> WorkflowState:
        evidence = state.get("evidence_chunks", [])
        context = state.get("context_text", "")
        query = state["query"]
        history = state.get("chat_history", [])
        doc_scope = state.get("document_scope")
        is_streaming = state.get("is_streaming", False)

        messages = self.writer_agent.build_messages(
            query=query,
            chat_history=history,
            results=evidence,
            context=context,
            document_scope=doc_scope,
            is_full_summary=(state.get("planner_intent") == "full_doc"),
        )

        if not is_streaming:
            draft = self.writer_agent.write_answer(messages)
        else:
            buffer = []
            streaming_active = False
            for token in self.writer_agent.write_answer_stream(messages):
                if not token:
                    continue
                if streaming_active:
                    writer(token)
                    buffer.append(token)
                else:
                    buffer.append(token)
                    accumulated = "".join(buffer)
                    # Stream immediately once a citation starts or after 6 tokens
                    if "[S" in accumulated or len(buffer) >= 6:
                        streaming_active = True
                        for t in buffer:
                            writer(t)
            draft = "".join(buffer)

        normalized = self.citation_service.normalize_citations(draft, evidence)

        trace = list(state.get("trace_steps", []))
        trace.append("answer_writer")
        details = dict(state.get("trace_details", {}))
        details["answer_writer"] = {"draft_length": len(draft)}

        return {
            "draft_answer": draft,
            "current_answer": normalized,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _citation_validator_node(self, state: WorkflowState, writer: StreamWriter) -> WorkflowState:
        current_ans = state.get("current_answer", "")
        evidence = state.get("evidence_chunks", [])
        repair_count = state.get("repair_count", 0)

        # Audit using CitationAgent's exposed verification tools
        audit_res = self.citation_agent.audit(current_ans, evidence)
        is_valid = audit_res.get("is_valid", False)
        error = audit_res.get("error")

        trace = list(state.get("trace_steps", []))
        trace.append("citation_validator")
        details = dict(state.get("trace_details", {}))
        details["citation_validator"] = {
            "tools_used": [t.name for t in self.citation_agent.tools],
        }

        if is_valid:
            details["citation_validator"]["status"] = "valid"
            return {
                "verified_answer": current_ans,
                "citation_error": None,
                "trace_steps": trace,
                "trace_details": details,
            }

        if repair_count < 1:
            details["citation_validator"]["status"] = "invalid"
            details["citation_validator"]["error"] = error
            return {
                "citation_error": error,
                "trace_steps": trace,
                "trace_details": details,
            }

        # Repair already attempted; use fallback tool if available
        fallback = self.citation_agent.extract_fallback(evidence)
        if fallback:
            details["citation_validator"]["status"] = "fallback_applied"
            if state.get("is_streaming", False):
                writer(fallback)
            return {
                "verified_answer": fallback,
                "citation_error": None,
                "trace_steps": trace,
                "trace_details": details,
            }

        bibliography = self.citation_agent.format_bibliography(evidence)
        if evidence and "**Sources:**" not in current_ans:
            current_ans = f"{current_ans.strip()}{bibliography}"
            if state.get("is_streaming", False):
                writer(bibliography)

        details["citation_validator"]["status"] = "appended_sources"
        return {
            "verified_answer": current_ans,
            "citation_error": None,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _route_after_validation(self, state: WorkflowState) -> str:
        if state.get("citation_error") and state.get("repair_count", 0) < 1:
            return "citation_repair"
        return "end"

    def _citation_repair_node(self, state: WorkflowState, writer: StreamWriter) -> WorkflowState:
        evidence = state.get("evidence_chunks", [])
        context = state.get("context_text", "")
        query = state["query"]
        history = state.get("chat_history", [])
        doc_scope = state.get("document_scope")
        current_ans = state.get("current_answer", "")
        is_streaming = state.get("is_streaming", False)

        messages = self.writer_agent.build_messages(
            query=query,
            chat_history=history,
            results=evidence,
            context=context,
            document_scope=doc_scope,
        )

        repaired = self.citation_agent.repair_answer(messages, current_ans, evidence)
        normalized = self.citation_service.normalize_citations(repaired, evidence)
        if is_streaming and "[S" in normalized:
            writer(normalized)

        trace = list(state.get("trace_steps", []))
        trace.append("citation_repair")
        details = dict(state.get("trace_details", {}))
        details["citation_repair"] = {"repair_attempt": state.get("repair_count", 0) + 1}

        return {
            "current_answer": normalized,
            "repair_count": state.get("repair_count", 0) + 1,
            "trace_steps": trace,
            "trace_details": details,
        }

    def _build_graph(self):
        builder = StateGraph(WorkflowState)

        # Register specialized agent nodes
        builder.add_node("planner", self._planner_node)
        builder.add_node("retrieval", self._retrieval_node)
        builder.add_node("table", self._table_node)
        builder.add_node("no_evidence", self._no_evidence_node)
        builder.add_node("answer_writer", self._answer_writer_node)
        builder.add_node("citation_validator", self._citation_validator_node)
        builder.add_node("citation_repair", self._citation_repair_node)

        if self.enable_evidence_analyst:
            builder.add_node("evidence_analyst", self._evidence_analyst_node)

        # Add Edges
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "retrieval")

        routing_map = {
            "no_evidence": "no_evidence",
            "table": "table",
            "answer_writer": "answer_writer",
        }
        if self.enable_evidence_analyst:
            routing_map["evidence_analyst"] = "evidence_analyst"

        builder.add_conditional_edges(
            "retrieval",
            self._route_after_retrieval,
            routing_map,
        )

        if self.enable_evidence_analyst:
            builder.add_conditional_edges(
                "evidence_analyst",
                self._route_after_evidence_analyst,
                {
                    "table": "table",
                    "answer_writer": "answer_writer",
                },
            )

        builder.add_edge("no_evidence", END)
        builder.add_edge("table", END)
        builder.add_edge("answer_writer", "citation_validator")

        builder.add_conditional_edges(
            "citation_validator",
            self._route_after_validation,
            {
                "citation_repair": "citation_repair",
                "end": END,
            },
        )

        builder.add_edge("citation_repair", "citation_validator")

        # Compile with MemorySaver checkpointer for stateful agent memory
        return builder.compile(checkpointer=self.checkpointer)

    def run(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        document_scope: Optional[str] = None,
        top_k: int = 10,
        session_id: Optional[str] = None,
    ) -> AgentQueryResult:
        """Synchronously execute the full multiagent workflow with checkpoint memory."""
        initial_state: WorkflowState = {
            "query": query,
            "chat_history": chat_history or [],
            "document_scope": document_scope,
            "top_k": top_k,
            "session_id": session_id,
            "trace_steps": [],
            "trace_details": {},
            "repair_count": 0,
            "expansion_count": 0,
            "is_streaming": False,
        }

        # Config with session thread_id for checkpoint isolation
        config = {"configurable": {"thread_id": session_id or "default_session"}}

        final_state = self.graph.invoke(initial_state, config=config)

        answer = final_state.get("verified_answer") or final_state.get("current_answer") or ""
        sources = final_state.get("evidence_chunks") or []
        citation_error = final_state.get("citation_error")
        is_table = "table_analyst" in final_state.get("trace_steps", [])

        self._last_result = AgentQueryResult(
            answer=answer,
            sources=sources,
            citation_error=citation_error,
            is_table_result=is_table,
            trace={
                "steps": final_state.get("trace_steps", []),
                "details": final_state.get("trace_details", {}),
            },
        )
        return self._last_result

    def stream(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        document_scope: Optional[str] = None,
        top_k: int = 10,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Execute the multiagent workflow with token-by-token streaming."""
        initial_state: WorkflowState = {
            "query": query,
            "chat_history": chat_history or [],
            "document_scope": document_scope,
            "top_k": top_k,
            "session_id": session_id,
            "trace_steps": [],
            "trace_details": {},
            "repair_count": 0,
            "expansion_count": 0,
            "is_streaming": True,
        }

        config = {"configurable": {"thread_id": session_id or "default_session"}}

        for chunk in self.graph.stream(initial_state, config=config, stream_mode="custom"):
            if chunk:
                yield str(chunk)

        final_state = self.graph.get_state(config).values
        answer = final_state.get("verified_answer") or final_state.get("current_answer") or ""
        sources = final_state.get("evidence_chunks") or []
        citation_error = final_state.get("citation_error")
        is_table = "table_analyst" in final_state.get("trace_steps", [])

        self._last_result = AgentQueryResult(
            answer=answer,
            sources=sources,
            citation_error=citation_error,
            is_table_result=is_table,
            trace={
                "steps": final_state.get("trace_steps", []),
                "details": final_state.get("trace_details", {}),
            },
        )
