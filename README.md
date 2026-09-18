# Semantic Caching Proxy for LLM APIs

A drop-in proxy that sits in front of any LLM API and caches *semantically
similar* queries — not just exact string matches — using local embeddings
and a FAISS similarity index. Includes a live dashboard showing cache hit
rate, latency saved, and estimated dollar savings.

Runs entirely on free tiers / local compute. No paid infra required.

## Why this exists

Most real-world LLM traffic is repetitive in meaning even when the wording
differs ("How do I reset my password?" vs "I forgot my password, how do I
change it?"). An exact-match cache misses this. This proxy catches it by
embedding every query and checking cosine similarity against everything
seen before.

## Architecture

```
client ──> FastAPI proxy ──> [cache miss] ──> LLM API (Groq / Ollama)
              │
              └──> [cache hit] ──> return cached response instantly, $0 cost
```

- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`), runs locally
  on CPU — no API calls, no cost.
- **Similarity search**: FAISS `IndexFlatIP` (inner product on normalized
  vectors = cosine similarity), persisted to disk.
- **LLM backend**: pluggable — Groq's free tier (real API, real $ math) or
  Ollama (fully local, zero cost, zero rate limits).
- **Dashboard**: single static HTML file, no build step, polls `/stats` and
  `/recent` every few seconds.

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env:
#   - LLM_PROVIDER=groq and GROQ_API_KEY=<your free key from console.groq.com>
#   - OR LLM_PROVIDER=ollama if you have Ollama running locally

uvicorn main:app --reload --port 8000
```

First run will download the embedding model (~80MB, one-time, free).

### 2. Dashboard

Just open `dashboard/index.html` directly in a browser (no server needed —
it talks to `http://localhost:8000` via CORS). Or serve it with any static
file server if you prefer.

### 3. Send some traffic

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset my password?"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I forgot my password, how do I change it?"}'
```

The second call should come back as a `"cache_status": "hit"` almost
instantly — that's the whole project working.

### 4. Run the threshold eval (the differentiator)

```bash
python eval_threshold.py
```

This measures true-hit-rate vs false-positive-rate across several
similarity thresholds on a small hand-labeled eval set. This number is
worth more in an interview than any amount of screenshots — it shows you
understand the *tradeoff*, not just the implementation.

## Talking points for interviews

- **Why inner-product FAISS index instead of L2?** Normalized vectors +
  inner product = cosine similarity, and `IndexFlatIP` is simpler/faster
  than computing L2 distance and converting.
- **Threshold tuning tradeoff**: too loose → wrong answers served for
  different questions (false cache hit); too tight → low hit rate, no
  savings. `eval_threshold.py` quantifies this instead of guessing.
- **Cache invalidation**: TTL-based lazy expiry (checked at lookup time,
  no background scheduler needed). Real production version would need
  invalidation hooks tied to source-data changes.
- **Multi-tenant safety**: current version is a shared cache — fine for a
  single knowledge base, unsafe if different users' answers shouldn't
  cross-contaminate. Would add a `tenant_id` namespace to the FAISS
  metadata and filter searches by it.
- **Cost model**: dashboard estimates savings using a configurable
  $-per-1M-token rate (`config.py`), applied to the tokens of the *skipped*
  LLM call. With Groq's real free-tier API, these are genuine dollar
  figures, not simulated ones.
- **Scaling past a demo**: FAISS `IndexFlatIP` is exact but O(n) per
  search. At real scale you'd swap to an approximate index (HNSW, IVF) and
  shard by tenant/namespace.

## Project structure

```
semantic-cache-proxy/
├── backend/
│   ├── main.py            # FastAPI app: /chat, /stats, /recent, /health
│   ├── cache.py            # SemanticCache: FAISS + sentence-transformers
│   ├── llm_client.py        # Groq / Ollama client
│   ├── config.py            # All tunables, env-driven
│   ├── eval_threshold.py     # Threshold eval script (interview gold)
│   ├── requirements.txt
│   └── .env.example
└── dashboard/
    └── index.html         # No-build-step live dashboard
```

## Free-tier notes

- Groq free tier: generous rate limits, real hosted models, no credit card.
- Ollama: 100% local and offline, zero rate limits, but needs the model
  pulled once (`ollama pull llama3.1:8b`) and enough RAM to run it.
- Embeddings and vector search never touch a paid API in this setup.
