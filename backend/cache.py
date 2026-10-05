"""
SemanticCache: Multi-tenant, namespace-isolated semantic caching engine.

Key Features & Engineering Rationale:
1. FAISS IndexIDMap2 + IndexFlatIP:
   - Uses Inner Product on normalized embeddings (cosine similarity).
   - Wrapped in IndexIDMap2 so vectors are bound to explicit 64-bit integer IDs,
     enabling point deletions when entries expire or undergo LRU eviction.

2. Multi-Tenant Namespace Isolation:
   - Queries are scoped by SHA256(model, system_prompt, temperature, namespace).
   - Prevents cross-contamination (e.g. pirate persona vs formal assistant).

3. High-Performance SQLite Metadata Storage:
   - Replaced full-JSON file dumps with atomic SQLite single-row updates/inserts.
   - Fast, thread-safe, and ACID compliant.

4. Dual Invalidation Strategy:
   - TTL Expiry: Lazy cleanup of stale entries during search lookups and stats sweeps.
   - LRU Eviction: When cache size exceeds MAX_CACHE_SIZE, the least-recently-accessed
     entry is purged from memory, SQLite, and FAISS.
"""

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


def compute_namespace_hash(
    model: str = "default",
    system_prompt: str = "",
    temperature: float = 0.7,
    namespace: str = "default",
) -> str:
    """Compute deterministic SHA256 hash for namespace scoping."""
    raw = f"{model.strip().lower()}|{system_prompt.strip()}|{temperature:.2f}|{namespace.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    id: int
    namespace_hash: str
    query: str
    response: str
    created_at: float
    last_accessed_at: float
    input_tokens_est: int
    output_tokens_est: int
    hit_count: int = 0


class SemanticCache:
    def __init__(self):
        os.makedirs(os.path.dirname(config.CACHE_INDEX_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(config.CACHE_DB_PATH), exist_ok=True)

        self.lock = threading.Lock()
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        self.db_path = config.CACHE_DB_PATH
        self._init_db()

        self.metadata: dict[int, CacheEntry] = {}
        self.next_id = 0

        # Initialize FAISS IndexIDMap2 wrapping IndexFlatIP
        flat_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        if os.path.exists(config.CACHE_INDEX_PATH):
            try:
                loaded_index = faiss.read_index(config.CACHE_INDEX_PATH)
                if isinstance(loaded_index, faiss.IndexIDMap2):
                    self.index = loaded_index
                else:
                    self.index = faiss.IndexIDMap2(loaded_index)
            except Exception:
                self.index = faiss.IndexIDMap2(flat_index)
        else:
            self.index = faiss.IndexIDMap2(flat_index)

        self._load_metadata_from_db()

    # ---------- Database Initialization ----------
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        id INTEGER PRIMARY KEY,
                        namespace_hash TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_accessed_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        input_tokens_est INTEGER NOT NULL,
                        output_tokens_est INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ns ON cache_entries(namespace_hash)"
                )
                conn.commit()

    def _load_metadata_from_db(self):
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM cache_entries")
                for row in cursor.fetchall():
                    entry = CacheEntry(
                        id=row["id"],
                        namespace_hash=row["namespace_hash"],
                        query=row["query"],
                        response=row["response"],
                        created_at=row["created_at"],
                        last_accessed_at=row["last_accessed_at"],
                        hit_count=row["hit_count"],
                        input_tokens_est=row["input_tokens_est"],
                        output_tokens_est=row["output_tokens_est"],
                    )
                    self.metadata[entry.id] = entry

            if self.metadata:
                self.next_id = max(self.metadata.keys()) + 1
            else:
                self.next_id = 0

    def _save_index(self):
        faiss.write_index(self.index, config.CACHE_INDEX_PATH)

    # ---------- Vector & Helper Utilities ----------
    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.array(vec, dtype="float32")

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // config.CHARS_PER_TOKEN)

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) > config.CACHE_TTL_SECONDS

    def _delete_entry(self, entry_id: int):
        """Helper to remove entry from metadata dict, SQLite, and FAISS vector index."""
        if entry_id in self.metadata:
            del self.metadata[entry_id]

        # Remove from SQLite DB
        with self._get_connection() as conn:
            conn.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))
            conn.commit()

        # Remove from FAISS index
        try:
            self.index.remove_ids(np.array([entry_id], dtype=np.int64))
            self._save_index()
        except Exception:
            pass

    # ---------- Public API ----------
    def lookup(
        self,
        query: str,
        model: str = "default",
        system_prompt: str = "",
        temperature: float = 0.7,
        namespace: str = "default",
    ) -> Optional[dict]:
        """Lookup cached response matching exact namespace and similarity threshold."""
        with self.lock:
            if self.index.ntotal == 0:
                return None

            ns_hash = compute_namespace_hash(model, system_prompt, temperature, namespace)
            query_vec = self._embed(query)
            k = min(10, self.index.ntotal)
            scores, ids = self.index.search(query_vec, k)

            for score, idx in zip(scores[0], ids[0]):
                if idx == -1:
                    continue

                entry_id = int(idx)
                entry = self.metadata.get(entry_id)
                if entry is None:
                    continue

                # 1. Check TTL Expiry - Lazy deletion
                if self._is_expired(entry):
                    self._delete_entry(entry_id)
                    continue

                # 2. Check Namespace Scoping
                if entry.namespace_hash != ns_hash:
                    continue

                # 3. Check Similarity Threshold
                if score >= config.SIMILARITY_THRESHOLD:
                    now = time.time()
                    entry.hit_count += 1
                    entry.last_accessed_at = now

                    # Efficient atomic single-row update in SQLite
                    with self._get_connection() as conn:
                        conn.execute(
                            "UPDATE cache_entries SET hit_count = ?, last_accessed_at = ? WHERE id = ?",
                            (entry.hit_count, now, entry_id),
                        )
                        conn.commit()

                    return {
                        "response": entry.response,
                        "similarity": float(score),
                        "matched_query": entry.query,
                        "input_tokens_saved": entry.input_tokens_est,
                        "output_tokens_saved": entry.output_tokens_est,
                    }

            return None

    def store(
        self,
        query: str,
        response: str,
        model: str = "default",
        system_prompt: str = "",
        temperature: float = 0.7,
        namespace: str = "default",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        """Store new query/response pair with LRU capacity enforcement."""
        with self.lock:
            now = time.time()
            ns_hash = compute_namespace_hash(model, system_prompt, temperature, namespace)

            # Evict expired entries or enforce LRU Eviction if cache is full
            active_ids = [eid for eid, e in self.metadata.items() if not self._is_expired(e)]

            # Clean expired items
            for eid in list(self.metadata.keys()):
                if self._is_expired(self.metadata[eid]):
                    self._delete_entry(eid)

            active_ids = list(self.metadata.keys())
            if len(active_ids) >= config.MAX_CACHE_SIZE:
                # Find Least Recently Used (LRU) entry
                lru_id = min(active_ids, key=lambda eid: self.metadata[eid].last_accessed_at)
                self._delete_entry(lru_id)

            entry_id = self.next_id
            self.next_id += 1

            in_tokens = input_tokens if input_tokens is not None else self._estimate_tokens(query)
            out_tokens = output_tokens if output_tokens is not None else self._estimate_tokens(response)

            entry = CacheEntry(
                id=entry_id,
                namespace_hash=ns_hash,
                query=query,
                response=response,
                created_at=now,
                last_accessed_at=now,
                input_tokens_est=in_tokens,
                output_tokens_est=out_tokens,
                hit_count=0,
            )

            # Store in SQLite
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO cache_entries
                    (id, namespace_hash, query, response, created_at, last_accessed_at, hit_count, input_tokens_est, output_tokens_est)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        entry.namespace_hash,
                        entry.query,
                        entry.response,
                        entry.created_at,
                        entry.last_accessed_at,
                        entry.hit_count,
                        entry.input_tokens_est,
                        entry.output_tokens_est,
                    ),
                )
                conn.commit()

            # Store in FAISS
            vec = self._embed(query)
            self.index.add_with_ids(vec, np.array([entry_id], dtype=np.int64))
            self._save_index()

            self.metadata[entry_id] = entry

    def stats(self) -> dict:
        with self.lock:
            active = [e for e in self.metadata.values() if not self._is_expired(e)]
            total_hits = sum(e.hit_count for e in active)
            return {
                "total_cached_queries": len(active),
                "total_cache_hits": total_hits,
                "index_size": self.index.ntotal,
                "max_cache_size": config.MAX_CACHE_SIZE,
                "ttl_seconds": config.CACHE_TTL_SECONDS,
            }
