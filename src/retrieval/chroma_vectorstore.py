"""
src/chroma_vectorstore.py

Backward-compatibility class.

Composes DocumentStore (ingestion/persistence) and HybridSearchEngine (hybrid search/reranking)
via multiple inheritance so all public and internal interfaces expected by existing tests
and pipelines remain 100% functional.
"""

from __future__ import annotations

import os
from typing import List, Any, Optional

from sentence_transformers import SentenceTransformer

from src.retrieval.document_store import DocumentStore
from src.retrieval.hybrid_search import HybridSearchEngine

CHROMA_DIR = "data/vector_store"
EMBEDDING_DIM = 384
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ChromaVectorStore(DocumentStore, HybridSearchEngine):
    """
    Unified vector store interface combining DocumentStore and HybridSearchEngine.
    Maintains full backward-compatibility with all existing methods and attributes.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        DocumentStore.__init__(
            self,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        HybridSearchEngine.__init__(
            self,
            chroma_client=self.client,
            bm25_path=self.bm25_path,
            embedding_model=embedding_model,
            model=self.model,
        )
        print(f"[ChromaVectorStore] Ready. Persisting at: {persist_dir}")
