"""
src/retrieval/hybrid_search.py

Hybrid retrieval engine: dense vector (ChromaDB) + BM25 (SQLite FTS5) fusion,
followed by cross-encoder reranking and neighbour-chunk expansion.
This module contains only search/retrieval logic — no ingestion or persistence.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, List, Optional

from sentence_transformers import SentenceTransformer

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb is required. Run: pip install chromadb")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


def _normalize_content_type(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "table":
        return "table"
    if normalized in {"text", "prose", "paragraph", "body", ""}:
        return "text"
    return "text"


class HybridSearchEngine:
    """
    Hybrid dense+sparse search engine backed by ChromaDB and SQLite BM25.

    Responsible for:
    - Dense vector retrieval via ChromaDB per-file collections.
    - Sparse BM25 retrieval via SQLite FTS5.
    - Reciprocal Rank Fusion (RRF) score combination.
    - Cross-encoder neural reranking.
    - Neighbour-chunk expansion for context completeness.
    - Page-range and section-keyword filtered lookups.
    """

    def __init__(
        self,
        chroma_client: Any,
        bm25_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        model: Optional[SentenceTransformer] = None,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.client = chroma_client
        self.bm25_path = bm25_path
        self.model = model or SentenceTransformer(embedding_model)
        self.reranker_model_name = reranker_model

        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(reranker_model)
        except Exception:
            self._reranker = None

    # ── BM25 helpers ───────────────────────────────────────────────────────────

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
    def _collection_name(filename: str) -> str:
        import os
        name = os.path.splitext(filename)[0]
        name = re.sub(r"[^a-zA-Z0-9\-]", "-", name)
        name = re.sub(r"-+", "-", name).strip("-")
        name = name[:63]
        if len(name) < 3:
            name = name.ljust(3, "0")
        return name.lower()

    def _bm25_candidates(
        self,
        query_text: str,
        candidate_k: int,
        collection_id: Optional[str],
    ) -> List[dict]:
        stop_words = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
            "is", "are", "be", "been", "by", "as", "with", "can", "what", "which",
            "how", "under", "must", "if", "that", "this", "from", "it", "its",
            "according", "does", "do", "shall", "may",
        }
        all_terms = list(dict.fromkeys(re.findall(r"\w+", query_text, flags=re.UNICODE)))
        terms = [t for t in all_terms if t.lower() not in stop_words]
        if not terms:
            terms = all_terms
        if not terms:
            return []

        match_expression = " OR ".join(f'"{term}"' for term in terms)
        sql = (
            "SELECT collection_id, chunk_id, content, metadata_json "
            "FROM chunk_fts WHERE chunk_fts MATCH ?"
        )
        parameters: List[Any] = [match_expression]
        if collection_id:
            sql += " AND collection_id = ?"
            parameters.append(collection_id)
        sql += " ORDER BY bm25(chunk_fts) LIMIT ?"
        parameters.append(candidate_k)

        with self._bm25_connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        results = []
        for collection, chunk_id, content, metadata_json in rows:
            metadata = json.loads(metadata_json)
            results.append(
                {
                    **metadata,
                    "content": content,
                    "_result_id": f"{collection}:{chunk_id}",
                    "_collection_id": collection,
                }
            )
        return results

    # ── Reranking ──────────────────────────────────────────────────────────────

    def _rerank(self, query_text: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return []
        if len(candidates) == 1:
            single = dict(candidates[0])
            return [{**{k: v for k, v in single.items() if not k.startswith("_")}, "rerank_score": 1.0}]
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model_name)

        pairs = [(query_text, candidate["content"]) for candidate in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda result: float(result[1]),
            reverse=True,
        )
        return [
            {
                **{key: value for key, value in candidate.items() if not key.startswith("_")},
                "rerank_score": float(score),
            }
            for candidate, score in ranked
        ]

    # ── Collection scope helpers ───────────────────────────────────────────────

    def _collections_in_scope(self, filename: Optional[str]) -> List[Any]:
        target_name = self._collection_name(filename) if filename else None
        collections = []
        try:
            available_collections = self.client.list_collections()
        except Exception as e:
            print(f"[HybridSearchEngine] Error listing collections: {e}")
            return []

        for collection_info in available_collections:
            if target_name and collection_info.name != target_name:
                continue
            try:
                collection = self.client.get_collection(collection_info.name)
                if collection.count() > 0:
                    collections.append(collection)
            except Exception as e:
                print(f"[HybridSearchEngine] Warning: Skipping unreadable collection '{collection_info.name}': {e}")
                continue
        return collections

    # ── Neighbour expansion ────────────────────────────────────────────────────

    def _expand_neighbor_chunks(self, candidates: List[dict], max_results: int) -> List[dict]:
        """
        Preserve top_k reranked candidates, augmenting with contiguous sibling chunks
        on the same page so tables, sub-clauses, and headings are never fragmented.
        """
        if not candidates:
            return []

        primary_candidates = candidates[:max_results]
        seen_keys = {(c.get("filename"), c.get("chunk_index")) for c in primary_candidates}
        supplemental = []
        collection_cache: dict = {}

        for candidate in primary_candidates:
            chunk_idx = candidate.get("chunk_index")
            filename = candidate.get("filename")
            page_num = candidate.get("page_number")
            col_id = candidate.get("_collection_id") or (
                self._collection_name(filename) if filename else None
            )

            if chunk_idx is None or not col_id or page_num is None:
                continue

            try:
                if col_id not in collection_cache:
                    collection_cache[col_id] = self.client.get_collection(col_id)
                col = collection_cache[col_id]
                sibling_ids = []
                if chunk_idx > 0:
                    sibling_ids.append((chunk_idx - 1, f"chunk-{chunk_idx - 1}"))
                sibling_ids.append((chunk_idx + 1, f"chunk-{chunk_idx + 1}"))

                lookup_ids = [sid for _, sid in sibling_ids]
                s_res = col.get(ids=lookup_ids, include=["documents", "metadatas"])
                for sid, doc, meta in zip(
                    s_res.get("ids", []), s_res.get("documents", []), s_res.get("metadatas", [])
                ):
                    s_idx = meta.get("chunk_index")
                    s_page = meta.get("page_number")
                    s_key = (filename, s_idx)
                    if s_key not in seen_keys and s_page == page_num:
                        seen_keys.add(s_key)
                        normalized_meta = dict(meta or {})
                        if "content_type" in normalized_meta:
                            normalized_meta["content_type"] = _normalize_content_type(
                                normalized_meta["content_type"]
                            )
                        supplemental.append(
                            {
                                **normalized_meta,
                                "content": doc,
                                "filename": filename or col.name,
                                "rerank_score": candidate.get("rerank_score", 0.0),
                                "_result_id": f"{col_id}:{sid}",
                                "_collection_id": col_id,
                            }
                        )
            except Exception:
                continue

        return primary_candidates + supplemental[:3]

    # ── Core hybrid query ──────────────────────────────────────────────────────

    def _hybrid_query(
        self,
        query_text: str,
        top_k: int,
        filename: Optional[str] = None,
    ) -> List[dict]:
        if top_k <= 0:
            return []

        collections = self._collections_in_scope(filename)
        total_chunks = sum(collection.count() for collection in collections)
        if total_chunks == 0:
            return []

        candidate_k = min(total_chunks, max(top_k * 4, 30))
        query_embedding = self.model.encode([query_text]).tolist()
        dense_candidates = []

        for collection in collections:
            try:
                count = collection.count()
                if count == 0:
                    continue
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(candidate_k, count),
                    include=["documents", "distances", "metadatas"],
                )
                collection_filename = (collection.metadata or {}).get("filename", collection.name)
                for chunk_id, content, distance, metadata in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                ):
                    normalized_metadata = dict(metadata or {})
                    if "content_type" in normalized_metadata:
                        normalized_metadata["content_type"] = _normalize_content_type(
                            normalized_metadata["content_type"]
                        )
                    dense_candidates.append(
                        {
                            **normalized_metadata,
                            "content": content,
                            "distance": distance,
                            "filename": collection_filename,
                            "_result_id": f"{collection.name}:{chunk_id}",
                            "_collection_id": collection.name,
                        }
                    )
            except Exception as e:
                print(f"[HybridSearchEngine] Warning: Could not query collection '{collection.name}': {e}")
                continue

        dense_candidates.sort(key=lambda result: result["distance"])
        dense_candidates = dense_candidates[:candidate_k]

        collection_id = self._collection_name(filename) if filename else None
        sparse_candidates = self._bm25_candidates(query_text, candidate_k, collection_id)

        # Reciprocal Rank Fusion
        fused: dict = {}
        for candidates in (dense_candidates, sparse_candidates):
            for rank, candidate in enumerate(candidates, start=1):
                result_id = candidate["_result_id"]
                if result_id not in fused:
                    fused[result_id] = {**candidate, "hybrid_score": 0.0}
                fused[result_id]["hybrid_score"] += 0.5 / (RRF_K + rank)

        fusion_ranked = sorted(
            fused.values(), key=lambda result: result["hybrid_score"], reverse=True
        )
        # Latency optimization: only rerank the top candidate pool instead of long tail
        top_candidates = fusion_ranked[:max(top_k + 4, 10)]
        reranked = self._rerank(query_text, top_candidates)
        return self._expand_neighbor_chunks(reranked, top_k)

    # ── Public search API ──────────────────────────────────────────────────────

    def query(self, query_text: str, filename: str, top_k: int = 5) -> List[dict]:
        """Return hybrid-search and reranked chunks from one document."""
        print(f"[HybridSearchEngine] Querying '{filename}' for: '{query_text}'")
        return self._hybrid_query(query_text, top_k, filename=filename)

    def query_all(self, query_text: str, top_k: int = 5) -> List[dict]:
        """Return hybrid-search and reranked chunks across all stored documents."""
        print(f"[HybridSearchEngine] Querying all stored files for: '{query_text}'")
        return self._hybrid_query(query_text, top_k)

    def query_by_pages(
        self, filename: str, start_page: int, end_page: int, max_chunks: int = 15
    ) -> List[dict]:
        """Retrieve sequential chunks within a specific page range (e.g. for TOC/Overview)."""
        collections = self._collections_in_scope(filename)
        if not collections:
            return []
        collection = collections[0]
        results = collection.get(include=["documents", "metadatas"])
        matched = []
        for content, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            page = meta.get("page_number", 0)
            if start_page <= page <= end_page:
                matched.append({**dict(meta), "content": content})
        matched.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))
        return matched[:max_chunks]

    def query_by_section(
        self, filename: str, section_keyword: str, max_chunks: int = 15
    ) -> List[dict]:
        """Retrieve chunks that belong to a specific section or chapter keyword."""
        collections = self._collections_in_scope(filename)
        if not collections:
            return []
        collection = collections[0]
        results = collection.get(include=["documents", "metadatas"])
        matched = []
        keyword_clean = section_keyword.casefold()
        for content, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            sec = str(meta.get("section", "")).casefold()
            content_lower = content.casefold()
            if keyword_clean in sec or keyword_clean in content_lower[:150]:
                matched.append({**dict(meta), "content": content})
        matched.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))
        return matched[:max_chunks]

    def get_document_outline(self, filename: str) -> dict:
        """
        Inspect document collection to extract structural overview and milestone chunks across all pages.
        Returns:
            {
                "filename": filename,
                "total_chunks": int,
                "page_count": int,
                "sections": List[str],
                "outline_chunks": List[dict],
            }
        """
        collections = self._collections_in_scope(filename)
        if not collections:
            return {"filename": filename, "total_chunks": 0, "page_count": 0, "sections": [], "outline_chunks": []}

        collection = collections[0]
        results = collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        total_chunks = len(docs)
        if total_chunks == 0:
            return {"filename": filename, "total_chunks": 0, "page_count": 0, "sections": [], "outline_chunks": []}

        all_items = []
        sections_seen: List[str] = []
        page_numbers: List[int] = []

        for content, meta in zip(docs, metas):
            meta_dict = dict(meta or {})
            p_num = meta_dict.get("page_number") or 1
            page_numbers.append(p_num)
            sec = meta_dict.get("section")
            if sec and sec not in sections_seen:
                sections_seen.append(sec)
            all_items.append({**meta_dict, "content": content})

        all_items.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))
        page_count = max(page_numbers) if page_numbers else 1
        meta_page_count = metas[0].get("page_count") if metas else None
        if meta_page_count and meta_page_count > page_count:
            page_count = meta_page_count

        # Case 1: Small document (<= 15 pages or <= 20 chunks) -> take all sequential chunks
        if page_count <= 15 or total_chunks <= 20:
            return {
                "filename": filename,
                "total_chunks": total_chunks,
                "page_count": page_count,
                "sections": sections_seen,
                "outline_chunks": all_items[:20],
            }

        # Case 2: Multi-page document (> 15 pages) -> assemble strategic anchors:
        # a) Preamble & TOC: first 3 pages
        head_chunks = [item for item in all_items if item.get("page_number", 0) <= 3][:5]
        # b) Concluding pages: last 2 pages
        tail_chunks = [item for item in all_items if item.get("page_number", 0) >= page_count - 1][:3]

        # c) Body milestones: sample evenly across page intervals
        step = max(page_count // 5, 2)
        milestone_pages = [step * i for i in range(1, 5) if step * i < page_count]
        milestone_chunks = []
        for target_p in milestone_pages:
            closest = min(all_items, key=lambda x: abs(x.get("page_number", 0) - target_p), default=None)
            if closest and closest not in head_chunks and closest not in tail_chunks and closest not in milestone_chunks:
                milestone_chunks.append(closest)

        # Assemble and deduplicate while maintaining document order
        selected_keys = set()
        outline_chunks = []
        for item in head_chunks + milestone_chunks + tail_chunks:
            key = (item.get("page_number"), item.get("chunk_index"))
            if key not in selected_keys:
                selected_keys.add(key)
                outline_chunks.append(item)

        outline_chunks.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_index", 0)))
        return {
            "filename": filename,
            "total_chunks": total_chunks,
            "page_count": page_count,
            "sections": sections_seen,
            "outline_chunks": outline_chunks[:18],
        }

