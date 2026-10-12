import unittest
from types import SimpleNamespace

from src.agents.research_agent import EvidenceSufficiencyAnalyst
from src.orchestration.graph import MultiAgentRAGGraph


class _FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    def __call__(self, messages):
        self.invocations.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "Synthesized answer [S1]."


class _FakeExpandingVectorStore:
    def __init__(self):
        self.queries = []

    def query(self, query_text, filename, top_k):
        self.queries.append(query_text)
        if "specifications" in query_text:
            return [{
                "source_id": "S2",
                "filename": filename,
                "page_number": 2,
                "content_type": "text",
                "rerank_score": 0.85,
                "content": "Expanded specifications for the building codes.",
            }]
        # First pass returns a low relevance chunk
        return [{
            "source_id": "S1",
            "filename": filename,
            "page_number": 1,
            "content_type": "text",
            "rerank_score": 0.05,
            "content": "Irrelevant introductory preamble.",
        }]

    def query_all(self, query_text, top_k):
        return self.query(query_text, "default.pdf", top_k)


class TestEvidenceSufficiencyAnalyst(unittest.TestCase):
    def test_sufficiency_check_identifies_strong_and_weak_evidence(self):
        analyst = EvidenceSufficiencyAnalyst()

        # Empty chunks -> insufficient
        self.assertFalse(analyst.is_evidence_sufficient("What is the rule?", []))

        # Chunk with high rerank score -> sufficient
        self.assertTrue(analyst.is_evidence_sufficient("What is the rule?", [
            {"content": "Any text", "rerank_score": 0.5}
        ]))

        # Table chunk -> sufficient
        self.assertTrue(analyst.is_evidence_sufficient("What is the rule?", [
            {"content": "Any text", "content_type": "table"}
        ]))

    def test_graph_executes_bounded_expansion_once(self):
        fake_vs = _FakeExpandingVectorStore()
        fake_llm = _FakeChatModel(["Answer based on evidence [S1]."])

        graph = MultiAgentRAGGraph(
            vectorstore=fake_vs,
            llm_invoker=fake_llm,
            enable_evidence_analyst=True,
        )

        result = graph.run(
            query="Solar panel requirements",
            document_scope="building.pdf",
        )

        # Confirm evidence_analyst participated
        self.assertIn("evidence_analyst", result.trace["steps"])
        # Confirm two queries occurred (initial + expanded)
        self.assertEqual(len(fake_vs.queries), 2)
        self.assertIn("specifications", fake_vs.queries[1])
        # Confirm sources were merged
        self.assertGreaterEqual(len(result.sources), 2)


if __name__ == "__main__":
    unittest.main()

