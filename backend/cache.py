"""
SemanticCache: the heart of the project.

Stores (query_embedding -> response) pairs in a FAISS index and answers
"have I seen something like this before?" using cosine similarity instead
of exact string matching.

Design notes (worth mentioning in interviews):
- We normalize embeddings and use an inner-product FAISS index, which is
  mathematically equivalent to cosine similarity on normalized vectors.
  This is a common trick to get cosine similarity search with FAISS's
  faster inner-product index type instead of a slower L2 + normalization.
- Metadata (the actual text + response + timestamps) lives in a plain JSON
  file for simplicity. In production this would be Postgres/Redis, but for
  a portfolio project this keeps the whole thing dependency-light and free.
- TTL-based invalidation is handled at lookup time (lazy expiry) rather than
  a background sweep, so there's no scheduler to run.
"""
import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


@dataclass
class CacheEntry:
    id: int
    query: str
    response: str
    created_at: float
    input_tokens_est: int
    output_tokens_est: int
    hit_count: int = 0


class SemanticCache:
    def __init__(self):
        os.makedirs(os.path.dirname(config.CACHE_INDEX_PATH), exist_ok=True)

        # Local embedding model -- no API calls, no cost, runs on CPU.
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        self.lock = threading.Lock()
        self.metadata: dict[int, CacheEntry] = {}
        self.next_id = 0

        if os.path.exists(config.CACHE_INDEX_PATH) and os.path.exists(config.CACHE_METADATA_PATH):
            self._load()
        else:
            self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)

    # ---------- persistence ----------
    def _load(self):
        self.index = faiss.read_index(config.CACHE_INDEX_PATH)
        with open(config.CACHE_METADATA_PATH, "r") as f:
            raw = json.load(f)
        self.metadata = {int(k): CacheEntry(**v) for k, v in raw.items()}
        self.next_id = (max(self.metadata.keys()) + 1) if self.metadata else 0

    def _save(self):
        faiss.write_index(self.index, config.CACHE_INDEX_PATH)
        with open(config.CACHE_METADATA_PATH, "w") as f:
            json.dump({k: asdict(v) for k, v in self.metadata.items()}, f)

    # ---------- helpers ----------
    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.array(vec, dtype="float32")

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // config.CHARS_PER_TOKEN)

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) > config.CACHE_TTL_SECONDS

    # ---------- public API ----------
    def lookup(self, query: str) -> Optional[dict]:
        """Return a cached entry if a semantically similar, non-expired
        query exists above the similarity threshold. Otherwise None."""
        with self.lock:
            if self.index.ntotal == 0:
                return None

            query_vec = self._embed(query)
            k = min(5, self.index.ntotal)
            scores, ids = self.index.search(query_vec, k)

            for score, idx in zip(scores[0], ids[0]):
                if idx == -1:
                    continue
                entry = self.metadata.get(int(idx))
                if entry is None or self._is_expired(entry):
                    continue
                if score >= config.SIMILARITY_THRESHOLD:
                    entry.hit_count += 1
                    self._save()
                    return {
                        "response": entry.response,
                        "similarity": float(score),
                        "matched_query": entry.query,
                        "input_tokens_saved": entry.input_tokens_est,
                        "output_tokens_saved": entry.output_tokens_est,
                    }
            return None

    def store(self, query: str, response: str):
        """Add a new query/response pair to the cache."""
        with self.lock:
            vec = self._embed(query)
            entry_id = self.next_id
            self.next_id += 1

            self.index.add(vec)
            self.metadata[entry_id] = CacheEntry(
                id=entry_id,
                query=query,
                response=response,
                created_at=time.time(),
                input_tokens_est=self._estimate_tokens(query),
                output_tokens_est=self._estimate_tokens(response),
            )
            self._save()

    def stats(self) -> dict:
        with self.lock:
            active = [e for e in self.metadata.values() if not self._is_expired(e)]
            total_hits = sum(e.hit_count for e in active)
            return {
                "total_cached_queries": len(active),
                "total_cache_hits": total_hits,
                "index_size": self.index.ntotal,
            }
