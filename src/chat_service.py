"""
src/chat_service.py

Application Service / Presentation Facade for the Streamlit UI and client applications.

Coordinates:
  - LLM client initialization and fallback failover (Gemini -> Groq)
  - Document ingestion through the retrieval layer
  - Supervised execution of the MultiAgentRAGGraph
  - Session-level caching of citations and execution traces
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from typing import Generator, List, Optional, Tuple

from dotenv import load_dotenv

# Suppress noisy Google GenAI AFC log spam
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.retrieval.document_store import DocumentStore
from src.retrieval.data_loader import load_all_documents
from src.retrieval.chroma_vectorstore import ChromaVectorStore
from src.orchestration.graph import MultiAgentRAGGraph
from src.services.citation_service import CitationService
from src.services.table_service import TableService

load_dotenv()

DEFAULT_LLM_MODEL = "gemini-flash-lite-latest"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
MAX_HISTORY_TURNS = 10
NO_EVIDENCE_MESSAGE = "The provided excerpt does not contain information regarding"
CITATION_FAILURE_MESSAGE = (
    "I couldn't provide a verified answer because the response did not cite "
    "valid retrieved sources. Please try rephrasing the question."
)


class ChatService:
    """
    Application service coordinating document ingestion, agent orchestration,
    and presentation-layer queries.

    Usage:
        service = ChatService()
        service.ingest(file_path, filename)
        for token in service.stream("question", chat_history):
            print(token, end="")
        sources = service.get_last_sources()
    """

    def __init__(self, llm_model: str = DEFAULT_LLM_MODEL, enable_multiagent: bool = True):
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        groq_api_key = os.getenv("GROQ_API_KEY", "")

        self.fallback_llm = None

        # Prefer Google Gemini (1M+ token context), with Groq fallback
        if gemini_api_key and ("gemini" in llm_model.lower() or not groq_api_key):
            actual_model = llm_model if "gemini" in llm_model.lower() else "gemini-flash-lite-latest"
            self.llm = ChatGoogleGenerativeAI(
                model=actual_model,
                google_api_key=gemini_api_key,
                temperature=0.2,
            )
            model_info = f"Google Gemini ({actual_model})"
            if groq_api_key:
                self.fallback_llm = ChatGroq(
                    groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b"
                )
        else:
            actual_model = (
                llm_model if "gemini" not in llm_model.lower() else "openai/gpt-oss-120b"
            )
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=actual_model)
            model_info = f"Groq ({actual_model})"
            if gemini_api_key:
                self.fallback_llm = ChatGoogleGenerativeAI(
                    model="gemini-flash-lite-latest",
                    google_api_key=gemini_api_key,
                    temperature=0.2,
                )

        self.vectorstore = ChromaVectorStore(
            embedding_model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.ingested_filename: Optional[str] = None
        self.chunk_count: int = 0
        self._last_sources: List[dict] = []
        self._last_citation_error: Optional[str] = None
        self._last_trace: dict = {}
        self.enable_multiagent = enable_multiagent

        # Instantiate shared services
        self.table_service = TableService()
        self.citation_service = CitationService()

        # Multi-Agent StateGraph
        self.multiagent_graph = MultiAgentRAGGraph(
            vectorstore=self.vectorstore,
            llm_invoker=self._invoke_llm,
            llm_streamer=self._stream_llm,
            table_service=self.table_service,
            citation_service=self.citation_service,
        )

        print(f"[RAGPipeline] Initialized with LLM: {model_info} (Multi-Agent: {self.enable_multiagent})")

    # ── LLM invocation ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text_content(content: object) -> str:
        """Ensure LLM response content is a clean string across all providers."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if hasattr(content, "text"):
            return str(content.text)
        return str(content or "")

    @staticmethod
    def _prepare_messages(messages: list) -> list:
        """
        Ensure messages have valid content and include at least one user turn for Gemini API.
        Prevents 'contents are required.' ClientError when only SystemMessage is sent.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        if not messages:
            return [HumanMessage(content="Please assist.")]

        has_human = any(
            isinstance(m, HumanMessage)
            or getattr(m, "type", "") == "human"
            or (isinstance(m, dict) and m.get("role") == "user")
            for m in messages
        )
        if not has_human:
            sanitized = []
            for m in messages:
                content = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
                if isinstance(m, SystemMessage) or (isinstance(m, dict) and m.get("role") == "system"):
                    sanitized.append(HumanMessage(content=str(content or "Please analyze.")))
                else:
                    sanitized.append(m)
            return sanitized

        return messages

    def _invoke_llm(self, messages: list) -> str:
        """Invoke primary LLM with automatic fallback on error."""
        messages = self._prepare_messages(messages)
        try:
            resp = self.llm.invoke(messages)
            return self._extract_text_content(resp.content)
        except Exception as primary_err:
            fallback = getattr(self, "fallback_llm", None)
            if fallback is not None:
                print(f"[RAGPipeline] Primary LLM failed ({primary_err}). Falling back...")
                resp = fallback.invoke(messages)
                return self._extract_text_content(resp.content)
            raise primary_err

    def _stream_llm(self, messages: list) -> Generator[str, None, None]:
        """Stream tokens from primary LLM with automatic fallback on failure."""
        messages = self._prepare_messages(messages)
        if hasattr(self.llm, "stream"):
            had_tokens = False
            try:
                for chunk in self.llm.stream(messages):
                    text = self._extract_text_content(chunk.content)
                    if text:
                        had_tokens = True
                        yield text
            except Exception as primary_err:
                fallback = getattr(self, "fallback_llm", None)
                if fallback is not None and not had_tokens:
                    print(f"[ChatService] Primary LLM streaming failed ({primary_err}). Falling back...")
                    if hasattr(fallback, "stream"):
                        try:
                            for chunk in fallback.stream(messages):
                                text = self._extract_text_content(chunk.content)
                                if text:
                                    yield text
                            return
                        except Exception:
                            pass
                    resp = fallback.invoke(messages)
                    text = self._extract_text_content(resp.content)
                    if text:
                        yield text
                elif not had_tokens:
                    resp = self._invoke_llm(messages)
                    if resp:
                        yield resp
                else:
                    raise primary_err
        else:
            yield self._invoke_llm(messages)

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def ingest(self, file_path: str, filename: str = "") -> int:
        """Load, chunk, embed, and persist a document into ChromaDB."""
        filename = filename or os.path.basename(file_path)
        print(f"[RAGPipeline] Ingesting: {filename}")

        tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        ext = os.path.splitext(file_path)[1].lower()
        dest_path = os.path.join(tmp_dir, f"uploaded{ext}")
        shutil.copy2(file_path, dest_path)

        try:
            documents = load_all_documents(tmp_dir)
            if not documents:
                raise ValueError("Could not extract any text from the uploaded file.")
            for document in documents:
                document.metadata["filename"] = filename
                document.metadata["source"] = filename
                table_context = document.metadata.get("table_context")
                if table_context:
                    context_lines = table_context.splitlines()
                    if context_lines and context_lines[0].startswith("Document: "):
                        context_lines[0] = f"Document: {filename}"
                    document.metadata["table_context"] = "\n".join(context_lines)
            count = self.vectorstore.build_from_documents(documents, filename=filename)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.ingested_filename = filename
        self.chunk_count = count
        return count

    # ── Graph resolution ───────────────────────────────────────────────────────

    def _get_or_create_graph(self) -> MultiAgentRAGGraph:
        graph = getattr(self, "multiagent_graph", None)
        if graph is not None:
            return graph

        vectorstore = getattr(self, "vectorstore", None)
        table_service = getattr(self, "table_service", None) or TableService()
        citation_service = getattr(self, "citation_service", None) or CitationService()

        # Invoker fallback for test doubles
        invoker = getattr(self, "_invoke_llm", None)
        if invoker is None:
            llm = getattr(self, "llm", None)
            if callable(llm):
                invoker = llm
            elif hasattr(llm, "invoke"):
                invoker = lambda msgs: self._extract_text_content(llm.invoke(msgs).content)
            else:
                invoker = lambda msgs: ""

        graph = MultiAgentRAGGraph(
            vectorstore=vectorstore,
            llm_invoker=invoker,
            llm_streamer=getattr(self, "_stream_llm", None),
            table_service=table_service,
            citation_service=citation_service,
        )
        self.multiagent_graph = graph
        return graph

    # ── Ask ────────────────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        top_k: int = TOP_K,
        filename: Optional[str] = None,
    ) -> str:
        """Execute question answering via the supervised MultiAgentRAGGraph."""
        active_scope = filename or getattr(self, "ingested_filename", None)
        if active_scope is None and not self.previously_ingested_files():
            self._last_citation_error = None
            return "⚠️ No document loaded. Please upload a file first."

        graph = self._get_or_create_graph()
        result = graph.run(
            query=query,
            chat_history=chat_history,
            document_scope=active_scope,
            top_k=top_k,
        )
        self._last_sources = result.sources
        self._last_citation_error = result.citation_error
        self._last_trace = result.trace
        return result.answer

    # ── Stream ─────────────────────────────────────────────────────────────────

    def stream(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        top_k: int = TOP_K,
        filename: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream the response token-by-token via the MultiAgentRAGGraph."""
        active_scope = filename or getattr(self, "ingested_filename", None)
        if active_scope is None and not self.previously_ingested_files():
            self._last_citation_error = None
            yield "⚠️ No document loaded. Please upload a file first."
            return

        graph = self._get_or_create_graph()
        for token in graph.stream(
            query=query,
            chat_history=chat_history,
            document_scope=active_scope,
            top_k=top_k,
        ):
            yield token

        last_result = getattr(graph, "_last_result", None)
        if last_result is not None:
            self._last_sources = last_result.sources
            self._last_citation_error = last_result.citation_error
            self._last_trace = last_result.trace

    def summarize_document(self, filename: Optional[str] = None) -> str:
        """
        Generate a complete whole-document executive summary using Hierarchical Map-Reduce.
        """
        target_filename = filename or getattr(self, "ingested_filename", None)
        if not target_filename:
            files = self.previously_ingested_files()
            if files:
                target_filename = files[0]
            else:
                return "⚠️ No document loaded. Please upload a file first."

        graph = self._get_or_create_graph()
        summary_agent = getattr(graph, "summary_agent", None)
        if summary_agent:
            return summary_agent.summarize_document(target_filename)

        return self.ask(f"Provide a comprehensive full document summary of {target_filename}", filename=target_filename)

    # ── Sources & Trace ────────────────────────────────────────────────────────

    def get_last_sources(self) -> List[dict]:
        """Return source chunks used in the last query."""
        return getattr(self, "_last_sources", [])

    def get_last_trace(self) -> dict:
        """Return multi-agent execution trace of the last query."""
        return getattr(self, "_last_trace", {})

    def get_last_citation_error(self) -> Optional[str]:
        """Return citation validation feedback for the last generated answer."""
        return getattr(self, "_last_citation_error", None)

    # ── Utility ───────────────────────────────────────────────────────────────

    def previously_ingested_files(self) -> List[str]:
        """Return all filenames stored in ChromaDB."""
        vs = getattr(self, "vectorstore", None)
        if vs and hasattr(vs, "list_files"):
            return vs.list_files()
        return []

    # ── Test & Backward Compatibility Delegations ──────────────────────────────

    def _build_messages(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        top_k: int = TOP_K,
        filename: Optional[str] = None,
    ) -> Tuple[list, List[dict], Optional[str]]:
        from src.agents.writer_agent import WriterAgent

        self._last_citation_error = None
        history = (chat_history or [])[-MAX_HISTORY_TURNS:]
        summary_mode, summary_target = self._detect_summary_intent(query)
        target_filename = filename or getattr(self, "ingested_filename", None)

        if summary_mode == "full_doc" and target_filename:
            results = self.vectorstore.query_by_pages(
                filename=target_filename,
                start_page=1,
                end_page=18,
                max_chunks=10,
            )
        elif summary_mode == "section" and summary_target and target_filename:
            results = self.vectorstore.query_by_section(
                filename=target_filename,
                section_keyword=summary_target,
                max_chunks=10,
            )
            if not results:
                results = self.vectorstore.query(
                    query_text=f"Summary of {summary_target}",
                    filename=target_filename,
                    top_k=top_k,
                )
        else:
            recent_history_for_search = (chat_history or [])[-2:]
            retrieval_query = (
                self._reformulate_query(query, recent_history_for_search)
                if recent_history_for_search
                else self._expand_legal_exclusion_query(query)
            )
            scope = filename or getattr(self, "ingested_filename", None)
            if scope:
                results = self.vectorstore.query(
                    query_text=retrieval_query,
                    filename=scope,
                    top_k=top_k,
                )
            else:
                results = self.vectorstore.query_all(
                    query_text=retrieval_query,
                    top_k=top_k,
                )

        context, results = WriterAgent.build_context_and_evidence(
            results,
            default_filename=target_filename,
            max_context_chars=60000,
        )
        self._last_sources = results
        document_scope = target_filename or "all stored documents"
        messages = WriterAgent.build_messages(
            query=query,
            chat_history=history,
            results=results,
            context=context,
            document_scope=document_scope,
        )

        has_table_results = any(
            result.get("content_type") == "table" for result in results
        )
        if has_table_results and self._has_table_aggregation_keyword(query):
            if self._has_explicit_table_aggregation_intent(query, results):
                table_answer = self._try_table_aggregation(query, results)
            else:
                table_answer = self._try_table_cell_lookup(query, results)
        else:
            table_answer = self._try_table_cell_lookup(query, results)

        return messages, results, table_answer

    @staticmethod
    def _normalize_table_text(value: str) -> str:
        return TableService.normalize_table_text(value)

    @staticmethod
    def _is_numeric_table_value(value: str) -> bool:
        return TableService.is_numeric_table_value(value)

    @staticmethod
    def _aggregation_operation(query: str) -> Optional[str]:
        return TableService.aggregation_operation(query)

    @classmethod
    def _has_table_aggregation_keyword(cls, query: str) -> bool:
        return TableService.has_table_aggregation_keyword(query)

    @classmethod
    def _has_explicit_table_aggregation_intent(cls, query: str, results: List[dict]) -> bool:
        return TableService.has_explicit_table_aggregation_intent(query, results)

    @staticmethod
    def _table_decimal(value: str):
        return TableService.table_decimal(value)

    @staticmethod
    def _format_decimal(value):
        return TableService.format_decimal(value)

    @classmethod
    def _format_fraction(cls, value):
        return TableService.format_fraction(value)

    def _try_table_aggregation(self, query: str, results: List[dict]) -> Optional[str]:
        svc = getattr(self, "table_service", None) or TableService()
        return svc.try_table_aggregation(query, results)

    def _try_table_cell_lookup(self, query: str, results: List[dict]) -> Optional[str]:
        svc = getattr(self, "table_service", None) or TableService()
        return svc.try_table_cell_lookup(query, results)

    @staticmethod
    def _normalize_citations(answer: str, results: List[dict]) -> str:
        return CitationService.normalize_citations(answer, results)

    @classmethod
    def _citation_validation_error(cls, answer: str, results: List[dict]) -> Optional[str]:
        return CitationService.citation_validation_error(answer, results)

    @classmethod
    def _format_source_references(cls, results: List[dict]) -> str:
        return CitationService.format_source_references(results)

    @classmethod
    def _cited_excerpt_fallback(cls, results: List[dict]) -> Optional[str]:
        return CitationService.cited_excerpt_fallback(results)

    @staticmethod
    def _expand_legal_exclusion_query(query: str) -> str:
        from src.agents.planner_agent import PlannerAgent
        return PlannerAgent.expand_legal_exclusion_query(query)

    @staticmethod
    def _detect_summary_intent(query: str) -> Tuple[str, Optional[str]]:
        from src.agents.planner_agent import PlannerAgent
        return PlannerAgent.detect_summary_intent(query)

    @staticmethod
    def _needs_reformulation(query: str) -> bool:
        from src.agents.planner_agent import PlannerAgent
        return PlannerAgent.needs_reformulation(query)

    def _reformulate_query(self, query: str, chat_history: List[Tuple[str, str]]) -> str:
        from src.agents.planner_agent import PlannerAgent
        invoker = getattr(self, "_invoke_llm", None)
        return PlannerAgent(llm_invoker=invoker).reformulate_query(query, chat_history)


# Backward-compatibility alias
RAGPipeline = ChatService

