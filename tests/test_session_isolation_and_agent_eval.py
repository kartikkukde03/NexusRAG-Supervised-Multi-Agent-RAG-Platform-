import unittest
from types import SimpleNamespace

from src.chat_service import ChatService as RAGPipeline
from src.orchestration.graph import MultiAgentRAGGraph


class _FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    def __call__(self, messages):
        self.invocations.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "Verified answer [S1]."

    def invoke(self, messages):
        return SimpleNamespace(content=self(messages))

    def stream(self, messages):
        yield SimpleNamespace(content=self(messages))


class _FakeMultiDocVectorStore:
    def __init__(self):
        self.call_history = []

    def query(self, query_text, filename, top_k):
        self.call_history.append(("query", query_text, filename))
        return [{
            "source_id": "S1",
            "filename": filename,
            "page_number": 1,
            "content_type": "text",
            "content": f"Content specifically from {filename} regarding {query_text}.",
        }]

    def query_all(self, query_text, top_k):
        self.call_history.append(("query_all", query_text, None))
        return [{
            "source_id": "S1",
            "filename": "global.pdf",
            "page_number": 1,
            "content_type": "text",
            "content": f"Global content regarding {query_text}.",
        }]

    def query_by_pages(self, filename, start_page, end_page, max_chunks):
        self.call_history.append(("query_by_pages", filename, start_page, end_page))
        return [{
            "source_id": "S1",
            "filename": filename,
            "page_number": start_page,
            "content_type": "text",
            "content": f"Overview of {filename} from page {start_page}.",
        }]

    def query_by_section(self, filename, section_keyword, max_chunks):
        self.call_history.append(("query_by_section", filename, section_keyword))
        return [{
            "source_id": "S1",
            "filename": filename,
            "page_number": 5,
            "content_type": "text",
            "content": f"Section {section_keyword} details for {filename}.",
        }]

    def list_files(self):
        return ["doc_A.pdf", "doc_B.pdf"]


class TestSessionIsolationAndAgentEval(unittest.TestCase):
    def test_concurrent_sessions_do_not_bleed_scope_or_sources(self):
        """Verify two independent sessions calling the same pipeline do not cross-contaminate."""
        pipeline = object.__new__(RAGPipeline)
        pipeline.enable_multiagent = True
        pipeline.vectorstore = _FakeMultiDocVectorStore()
        pipeline.llm = _FakeChatModel([
            "Answer for Session A [S1].",
            "Answer for Session B [S1].",
        ])
        pipeline.multiagent_graph = MultiAgentRAGGraph(
            vectorstore=pipeline.vectorstore,
            llm_invoker=pipeline.llm,
        )

        # Session A asks about doc_A.pdf
        ans_a = pipeline.ask("Question A", filename="doc_A.pdf")
        sources_a = list(pipeline.get_last_sources())

        # Session B asks about doc_B.pdf
        ans_b = pipeline.ask("Question B", filename="doc_B.pdf")
        sources_b = list(pipeline.get_last_sources())

        self.assertIn("Answer for Session A", ans_a)
        self.assertIn("Answer for Session B", ans_b)
        self.assertEqual(sources_a[0]["filename"], "doc_A.pdf")
        self.assertEqual(sources_b[0]["filename"], "doc_B.pdf")

        # Verify VectorStore received distinct document targets
        calls = pipeline.vectorstore.call_history
        self.assertEqual(calls[0], ("query", "Question A", "doc_A.pdf"))
        self.assertEqual(calls[1], ("query", "Question B", "doc_B.pdf"))

    def test_full_document_summary_routed_via_planner_and_pages(self):
        """Verify 'summarize this document' triggers query_by_pages in multiagent graph."""
        pipeline = object.__new__(RAGPipeline)
        pipeline.enable_multiagent = True
        pipeline.vectorstore = _FakeMultiDocVectorStore()
        pipeline.llm = _FakeChatModel(["Comprehensive document overview [S1]."])
        pipeline.multiagent_graph = MultiAgentRAGGraph(
            vectorstore=pipeline.vectorstore,
            llm_invoker=pipeline.llm,
        )

        ans = pipeline.ask("Please give me a summary of this document", filename="doc_A.pdf")
        self.assertIn("Comprehensive document overview [S1].", ans)
        calls = pipeline.vectorstore.call_history
        self.assertEqual(calls[0][0], "query_by_pages")
        self.assertEqual(calls[0][1], "doc_A.pdf")

    def test_section_summary_routed_via_planner_and_section(self):
        """Verify 'summarize chapter 4' triggers query_by_section in multiagent graph."""
        pipeline = object.__new__(RAGPipeline)
        pipeline.enable_multiagent = True
        pipeline.vectorstore = _FakeMultiDocVectorStore()
        pipeline.llm = _FakeChatModel(["Chapter 4 summary [S1]."])
        pipeline.multiagent_graph = MultiAgentRAGGraph(
            vectorstore=pipeline.vectorstore,
            llm_invoker=pipeline.llm,
        )

        ans = pipeline.ask("Summarize chapter 4", filename="doc_A.pdf")
        self.assertIn("Chapter 4 summary [S1].", ans)
        calls = pipeline.vectorstore.call_history
        self.assertEqual(calls[0][0], "query_by_section")
        self.assertEqual(calls[0][1], "doc_A.pdf")
        self.assertEqual(calls[0][2], "chapter 4")

    def test_stream_respects_explicit_filename(self):
        """Verify stream() generator passes explicit filename and produces stream output."""
        pipeline = object.__new__(RAGPipeline)
        pipeline.enable_multiagent = True
        pipeline.vectorstore = _FakeMultiDocVectorStore()
        pipeline.llm = _FakeChatModel(["Streamed answer [S1]."])

        tokens = list(pipeline.stream("What is the rule?", filename="doc_B.pdf"))
        full_text = "".join(tokens)
        self.assertIn("Streamed answer [S1].", full_text)
        self.assertEqual(pipeline.get_last_sources()[0]["filename"], "doc_B.pdf")


if __name__ == "__main__":
    unittest.main()

