import unittest
from types import SimpleNamespace

from src.orchestration.graph import MultiAgentRAGGraph


class _FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    def __call__(self, messages):
        self.invocations.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "Default answer."


class _FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_filename = None

    def query(self, query_text, filename, top_k):
        self.last_query = query_text
        self.last_filename = filename
        return [dict(r) for r in self.results[:top_k]]

    def query_all(self, query_text, top_k):
        self.last_query = query_text
        self.last_filename = None
        return [dict(r) for r in self.results[:top_k]]

    def query_by_pages(self, filename, start_page, end_page, max_chunks):
        return [dict(r) for r in self.results[:max_chunks]]

    def query_by_section(self, filename, section_keyword, max_chunks):
        return [dict(r) for r in self.results[:max_chunks]]


class TestMultiAgentRAGGraph(unittest.TestCase):
    def test_table_aggregation_executes_without_llm(self):
        table_result = {
            "source_id": "S1",
            "filename": "stats.pdf",
            "page_number": 10,
            "content_type": "table",
            "table_header_markdown": "| Region | Population |\n| --- | --- |",
            "content": "| Region | Population |\n| --- | --- |\n| North | 10 |\n| South | 20 |",
        }
        fake_vs = _FakeVectorStore([table_result])
        fake_llm = _FakeChatModel([])

        graph = MultiAgentRAGGraph(vectorstore=fake_vs, llm_invoker=fake_llm)
        result = graph.run(
            query="Sum population for North and South",
            document_scope="stats.pdf",
        )

        self.assertIn("Sum of Population for North (10) and South (20) = 30 [S1].", result.answer)
        self.assertTrue(result.is_table_result)
        self.assertEqual(len(fake_llm.invocations), 0)
        self.assertIn("table_analyst", result.trace["steps"])

    def test_answer_writer_validates_citations_successfully(self):
        text_result = {
            "filename": "doc.pdf",
            "page_number": 5,
            "content_type": "text",
            "content": "The standard height limit is 15 meters for residential buildings.",
        }
        fake_vs = _FakeVectorStore([text_result])
        fake_llm = _FakeChatModel(["The residential height limit is 15 meters [S1]."])

        graph = MultiAgentRAGGraph(vectorstore=fake_vs, llm_invoker=fake_llm)
        result = graph.run(
            query="What is the height limit?",
            document_scope="doc.pdf",
        )

        self.assertIn("15 meters [S1]", result.answer)
        self.assertIsNone(result.citation_error)
        self.assertIn("answer_writer", result.trace["steps"])
        self.assertIn("citation_validator", result.trace["steps"])

    def test_citation_repair_triggers_on_missing_citation_and_succeeds(self):
        text_result = {
            "filename": "doc.pdf",
            "page_number": 5,
            "content_type": "text",
            "content": "Solar panels must be placed on roofs.",
        }
        fake_vs = _FakeVectorStore([text_result])
        # First call: misses citation. Second call (repair): has [S1].
        fake_llm = _FakeChatModel([
            "Solar panels must be placed on roofs without any citation.",
            "Solar panels must be placed on roofs [S1].",
        ])

        graph = MultiAgentRAGGraph(vectorstore=fake_vs, llm_invoker=fake_llm)
        result = graph.run(
            query="Where to put solar panels?",
            document_scope="doc.pdf",
        )

        self.assertIn("[S1]", result.answer)
        self.assertIsNone(result.citation_error)
        self.assertIn("citation_repair", result.trace["steps"])
        self.assertEqual(len(fake_llm.invocations), 2)

    def test_empty_evidence_routes_to_no_evidence(self):
        fake_vs = _FakeVectorStore([])
        fake_llm = _FakeChatModel([])

        graph = MultiAgentRAGGraph(vectorstore=fake_vs, llm_invoker=fake_llm)
        result = graph.run(
            query="Unknown topic",
            document_scope="doc.pdf",
        )

        self.assertIn("does not contain information regarding Unknown topic", result.answer)
        self.assertIn("no_evidence", result.trace["steps"])


if __name__ == "__main__":
    unittest.main()

