"""
src/retrieval/document_store.py

Low-level ChromaDB persistence layer.
Handles document ingestion, BM25 index sync, collection management, and file listing.
This module contains only storage concerns — no search or retrieval logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, List

from sentence_transformers import SentenceTransformer

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb is required. Run: pip install chromadb")

from src.retrieval.embeddings import EmbeddingPipeline

CHROMA_DIR = "data/vector_store"


def _normalize_content_type(value: Any) -> str:
    """Canonicalize legacy content labels to the text/table contract used by the app."""
    normalized = str(value or "").strip().lower()
    if normalized == "table":
        return "table"
    if normalized in {"text", "prose", "paragraph", "body", ""}:
        return "text"
    return "text"


class DocumentStore:
    """
    ChromaDB-backed document store.

    Responsible for:
    - Ingesting (chunking, embedding, persisting) documents into per-file collections.
    - Maintaining a BM25 SQLite index in sync with Chroma collections.
    - Exposing file listing and chunk-count utilities.

    Not responsible for search/retrieval — see HybridSearchEngine.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.bm25_path = os.path.join(persist_dir, "bm25_index.sqlite3")

        self._initialize_bm25_index()
        self._sync_bm25_index()
        print(f"[DocumentStore] Ready. Persisting at: {persist_dir}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _collection_name(filename: str) -> str:
        """Convert a filename to a valid Chroma collection name (3-63 chars, alphanumeric + hyphens)."""
        import re
        name = os.path.splitext(filename)[0]
        name = re.sub(r"[^a-zA-Z0-9\-]", "-", name)
        name = re.sub(r"-+", "-", name).strip("-")
        name = name[:63]
        if len(name) < 3:
            name = name.ljust(3, "0")
        return name.lower()

    def _get_or_create_collection(self, filename: str):
        col_name = self._collection_name(filename)
        return self.client.get_or_create_collection(
            name=col_name,
            metadata={"filename": filename, "hnsw:space": "l2"},
        )

    def _initialize_bm25_index(self) -> None:
        with self._bm25_connection() as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
                "collection_id UNINDEXED, chunk_id UNINDEXED, content, "
                "metadata_json UNINDEXED, tokenize='unicode61')"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS bm25_state ("
                "collection_id TEXT PRIMARY KEY, chunk_count INTEGER NOT NULL, "
                "signature TEXT NOT NULL)"
            )

    @contextmanager
    def _bm25_connection(self):
        connection = sqlite3.connect(self.bm25_path, timeout=30)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _collection_signature(ids: List[str], documents: List[str], metadatas: List[dict]) -> str:
        records = sorted(zip(ids, documents, metadatas), key=lambda record: record[0])
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _replace_bm25_collection(
        self,
        collection_id: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[dict],
        signature: str,
    ) -> None:
        rows = [
            (
                collection_id,
                chunk_id,
                content,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            )
            for chunk_id, content, metadata in zip(ids, documents, metadatas)
        ]
        with self._bm25_connection() as connection:
            connection.execute(
                "DELETE FROM chunk_fts WHERE collection_id = ?",
                (collection_id,),
            )
            connection.executemany(
                "INSERT INTO chunk_fts "
                "(collection_id, chunk_id, content, metadata_json) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            connection.execute(
                "INSERT OR REPLACE INTO bm25_state "
                "(collection_id, chunk_count, signature) VALUES (?, ?, ?)",
                (collection_id, len(rows), signature),
            )

    @staticmethod
    def _set_collection_signature(collection: Any, signature: str) -> None:
        metadata = dict(collection.metadata or {})
        metadata.pop("hnsw:space", None)
        metadata["bm25_signature"] = signature
        collection.modify(metadata=metadata)

    def _sync_bm25_index(self) -> None:
        """Rebuild missing or outdated BM25 entries from the persistent Chroma index."""
        collections = list(self.client.list_collections())
        active_names = {collection_info.name for collection_info in collections}
        with self._bm25_connection() as connection:
            indexed_names = {
                row[0] for row in connection.execute("SELECT collection_id FROM bm25_state")
            }
            stale_names = indexed_names - active_names
            for collection_id in stale_names:
                connection.execute(
                    "DELETE FROM chunk_fts WHERE collection_id = ?",
                    (collection_id,),
                )
                connection.execute(
                    "DELETE FROM bm25_state WHERE collection_id = ?",
                    (collection_id,),
                )

        for collection_info in collections:
            collection = self.client.get_collection(collection_info.name)
            count = collection.count()
            current_signature = (collection.metadata or {}).get("bm25_signature", "")
            with self._bm25_connection() as connection:
                state = connection.execute(
                    "SELECT chunk_count, signature FROM bm25_state WHERE collection_id = ?",
                    (collection.name,),
                ).fetchone()
            if count == 0:
                self._replace_bm25_collection(collection.name, [], [], [], current_signature)
                continue
            if state and state == (count, current_signature) and current_signature:
                continue

            stored = collection.get(include=["documents", "metadatas"])
            ids = stored["ids"]
            documents = stored["documents"]
            metadatas = stored["metadatas"]
            signature = self._collection_signature(ids, documents, metadatas)
            self._replace_bm25_collection(collection.name, ids, documents, metadatas, signature)
            self._set_collection_signature(collection, signature)

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_from_documents(self, documents: List[Any], filename: str = "unknown") -> int:
        """
        Chunk, embed, and persist documents under a per-filename collection.
        Re-uploading the same file deletes the old collection first (clean upsert).

        Returns the number of chunks stored.
        """
        print(f"[DocumentStore] Building index for '{filename}'...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model=self.model,
        )
        chunks = emb_pipe.chunk_documents(documents)
        if not chunks:
            raise ValueError(f"No indexable text was extracted from '{filename}'.")
        embeddings = emb_pipe.embed_chunks(chunks)
        chunk_ids = [f"chunk-{i}" for i in range(len(chunks))]
        chunk_documents = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [
            {
                **{
                    key: _normalize_content_type(value) if key == "content_type" else value
                    for key, value in chunk.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                },
                "filename": filename,
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]
        signature = self._collection_signature(chunk_ids, chunk_documents, chunk_metadatas)

        # Delete old collection before writing new one
        self.delete_by_filename(filename)

        collection = self._get_or_create_collection(filename)
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            documents=chunk_documents,
            metadatas=chunk_metadatas,
        )
        self._set_collection_signature(collection, signature)
        self._replace_bm25_collection(
            collection.name, chunk_ids, chunk_documents, chunk_metadatas, signature
        )

        print(f"[DocumentStore] Stored {len(chunks)} chunks for '{filename}'.")
        return len(chunks)

    def delete_by_filename(self, filename: str) -> None:
        """Delete the ChromaDB collection and BM25 entries for this filename, if they exist."""
        col_name = self._collection_name(filename)
        existing_names = {collection.name for collection in self.client.list_collections()}
        if col_name in existing_names:
            self.client.delete_collection(col_name)
            print(f"[DocumentStore] Deleted old collection for '{filename}'.")
        with self._bm25_connection() as connection:
            connection.execute("DELETE FROM chunk_fts WHERE collection_id = ?", (col_name,))
            connection.execute("DELETE FROM bm25_state WHERE collection_id = ?", (col_name,))

    def list_files(self) -> List[str]:
        """Return display names of all stored documents (non-empty collections only)."""
        files = []
        try:
            collections = self.client.list_collections()
            for col in collections:
                try:
                    full_col = self.client.get_collection(col.name)
                    if full_col.count() == 0:
                        continue
                    meta = full_col.metadata or {}
                    filename = meta.get("filename", col.name)
                    files.append(filename)
                except Exception as inner_e:
                    print(f"[DocumentStore] Skipping problematic collection '{col.name}': {inner_e}")
                    continue
        except Exception as e:
            print(f"[DocumentStore] Error listing files: {e}")
        return sorted(files)

    def chunk_count(self, filename: str) -> int:
        """Return number of chunks stored for a given filename."""
        try:
            col = self._get_or_create_collection(filename)
            return col.count()
        except Exception:
            return 0
