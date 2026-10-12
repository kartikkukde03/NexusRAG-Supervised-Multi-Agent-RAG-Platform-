"""
src/retrieval/

Document ingestion, chunking, embedding, storage, and hybrid search.

Modules:
  - document_store.py  — ChromaDB persistence & SQLite BM25 synchronization
  - hybrid_search.py   — Dense + BM25 reciprocal rank fusion & cross-encoder reranking
  - data_loader.py     — Document loading (PDF, DOCX, etc.) & table parsing
  - embeddings.py      — Chunking & embedding pipeline
"""

from src.retrieval.document_store import DocumentStore
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.data_loader import load_all_documents
from src.retrieval.embeddings import EmbeddingPipeline
from src.retrieval.chroma_vectorstore import ChromaVectorStore

__all__ = [
    "DocumentStore",
    "HybridSearchEngine",
    "load_all_documents",
    "EmbeddingPipeline",
    "ChromaVectorStore",
]
