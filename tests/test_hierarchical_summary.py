import unittest
from types import SimpleNamespace

from src.agents.summary_agent import DocumentSummaryAgent
from src.orchestration.graph import MultiAgentRAGGraph


class _FakeChatModel:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.invocations = []

    def __call__(self, messages):
        self.invocations.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "Distilled section summary with key findings [S1]."


class _FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def get_all_document_chunks(self, filename: str):
        return [dict(c) for c in self.chunks if c.get("filename") == filename or not c.get("filename")]

    def query_by_pages(self, filename: str, start_page: int, end_page: int, max_chunks: int):
        return [dict(c) for c in self.chunks[:max_chunks]]

    def query(self, query_text: str, filename: str, top_k: int):
        return [dict(c) for c in self.chunks[:top_k]]


class TestHierarchicalDocumentSummary(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            {
                "filename": "regulations.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "section": "Chapter 1: Definitions",
                "content": "Rule 1: Building height shall not exceed 15 meters for standard residential zones.",
            },
            {
                "filename": "regulations.pdf",
                "page_number": 1,
                "chunk_index": 1,
                "section": "Chapter 1: Definitions",
                "content": "Rule 2: Setback requirements are fixed at 3 meters from property boundaries.",
            },
            {
                "filename": "regulations.pdf",
                "page_number": 5,
                "chunk_index": 2,
                "section": "Chapter 2: Commercial Standards",
                "content": "Rule 3: Commercial buildings require a minimum parking ratio of 2 spaces per 100 sq meters.",
            },
            {
                "filename": "regulations.pdf",
                "page_number": 8,
                "chunk_index": 3,
                "section": "Chapter 2: Commercial Standards",
                "content": "Rule 4: Maximum Floor Space Index (FSI) for commercial zones is 2.5.",
            },
            {
                "filename": "regulations.pdf",
                "page_number": 12,
                "chunk_index": 4,
                "section": "Chapter 3: Environmental Compliance",
                "content": "Rule 5: Rainwater harvesting is mandatory for all plots exceeding 500 square meters.",
            },
            {
                "filename": "regulations.pdf",
                "page_number": 15,
                "chunk_index": 5,
                "section": "Chapter 3: Environmental Compliance",
                "content": "Rule 6: Solar panel installation is incentivized with 5 percent rebate on municipal dues.",
            },
            {
                "filename": "regulations.pdf",
                "page_number": 20,
                "chunk_index": 6,
                "section": "Chapter 4: Penalties",
                "content": "Rule 7: Unauthorized construction incurs penalty up to 3 times standard fees.",
            },
        ]

    def test_partition_chunks_clusters_by_section_and_bounds(self):
        agent = DocumentSummaryAgent(vectorstore=None, llm_invoker=None)
        clusters = agent.partition_chunks(self.sample_chunks, max_chunks_per_cluster=4)

        self.assertGreater(len(clusters), 1)
        # Verify all chunks are accounted for across clusters
        total_partitioned = sum(len(c["chunks"]) for c in clusters)
        self.assertEqual(total_partitioned, len(self.sample_chunks))

        # Check section names and page bounds
        first_cluster = clusters[0]
        self.assertEqual(first_cluster["page_start"], 1)
        self.assertIn("Chapter 1", first_cluster["section_name"])

    def test_map_reduce_offline_heuristic(self):
        fake_vs = _FakeVectorStore(self.sample_chunks)
        agent = DocumentSummaryAgent(vectorstore=fake_vs, llm_invoker=None)

        summary = agent.summarize_document("regulations.pdf")
        self.assertIn("Executive Summary", summary)
        self.assertIn("regulations.pdf", summary)
        self.assertIn("Chapter 1", summary)

    def test_map_reduce_with_llm_and_caching(self):
        fake_vs = _FakeVectorStore(self.sample_chunks)
        # Mock responses: map phase returns bullet digests, reduce returns executive summary
        mock_llm = _FakeChatModel([
            "- Residential height limit 15m.\n- Setback 3m.",
            "- Commercial parking 2 spaces/100m².\n- FSI 2.5.",
            "- Rainwater harvesting mandatory over 500m².\n- Solar rebate 5%.",
            "- Penalties up to 3x fees.",
            "# Executive Summary\n\nFull synthesis of all regulations [S1] [S2].",
        ])
        agent = DocumentSummaryAgent(vectorstore=fake_vs, llm_invoker=mock_llm)

        summary = agent.summarize_document("regulations.pdf")
        self.assertIn("Executive Summary", summary)
        initial_invocations = len(mock_llm.invocations)
        self.assertGreater(initial_invocations, 0)

        # Second call must hit the in-memory cache with zero extra LLM invocations
        cached_summary = agent.summarize_document("regulations.pdf")
        self.assertEqual(summary, cached_summary)
        self.assertEqual(len(mock_llm.invocations), initial_invocations)

    def test_gather_full_document_evidence_spans_entire_document(self):
        fake_vs = _FakeVectorStore(self.sample_chunks)
        agent = DocumentSummaryAgent(vectorstore=fake_vs, llm_invoker=None)

        evidence = agent.gather_full_document_evidence("regulations.pdf")
        self.assertGreaterEqual(len(evidence), 2)
        # Evidence chunks must cover start and end pages
        pages = [e["page_number"] for e in evidence]
        self.assertIn(1, pages)
        self.assertTrue(any(p >= 12 for p in pages))

    def test_graph_routes_full_doc_summary_to_summary_agent(self):
        fake_vs = _FakeVectorStore(self.sample_chunks)
        fake_llm = _FakeChatModel([
            "Executive Overview of regulations: All rules covered [S1] [S2].",
        ])
        graph = MultiAgentRAGGraph(vectorstore=fake_vs, llm_invoker=fake_llm)

        result = graph.run(
            query="Provide a comprehensive whole document summary of regulations.pdf",
            document_scope="regulations.pdf",
        )

        self.assertIn("answer_writer", result.trace["steps"])
        self.assertIn("retrieval", result.trace["steps"])
        self.assertGreater(len(result.sources), 0)

    def test_adaptive_scaling_for_400_page_document(self):
        """Simulate a 400-page document with 800 chunks and verify it forms 8-14 clusters."""
        simulated_chunks = []
        for p in range(1, 401):
            simulated_chunks.append({
                "filename": "massive_code_400p.pdf",
                "page_number": p,
                "chunk_index": 0,
                "section": f"Chapter {(p // 40) + 1}",
                "content": f"Text from page {p} of the 400-page document.",
            })
            simulated_chunks.append({
                "filename": "massive_code_400p.pdf",
                "page_number": p,
                "chunk_index": 1,
                "section": f"Chapter {(p // 40) + 1}",
                "content": f"Additional content from page {p}.",
            })

        agent = DocumentSummaryAgent(vectorstore=None, llm_invoker=None)
        clusters = agent.partition_chunks(simulated_chunks)

        # Total clusters must be between 8 and 14 (safe for API rate limits and low latency)
        self.assertGreaterEqual(len(clusters), 8)
        self.assertLessEqual(len(clusters), 14)

        # Verify 100% chunk coverage
        total_covered = sum(len(c["chunks"]) for c in clusters)
        self.assertEqual(total_covered, 800)

        # Verify start and end span
        self.assertEqual(clusters[0]["page_start"], 1)
        self.assertEqual(clusters[-1]["page_end"], 400)


if __name__ == "__main__":
    unittest.main()

