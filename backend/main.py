"""
The proxy server itself.

POST /chat        -> the actual proxy endpoint clients call instead of
                      hitting the LLM API directly
GET  /stats        -> aggregate metrics for the dashboard
GET  /recent       -> recent request log (hit/miss feed) for the dashboard
GET  /health       -> basic health check
"""
import time
from collections import deque
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from cache import SemanticCache
from llm_client import call_llm, LLMError

app = FastAPI(title="Semantic Caching LLM Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a local portfolio demo; lock down in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = SemanticCache()

# In-memory ring buffer of recent requests, purely for the dashboard feed.
# Not persisted -- restart clears it. Fine for a demo; swap for a DB table
# if you want this to survive restarts.
recent_requests = deque(maxlen=200)


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    cache_status: Literal["hit", "miss"]
    latency_ms: float
    similarity: float | None = None


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * config.COST_PER_1M_INPUT_TOKENS
        + (output_tokens / 1_000_000) * config.COST_PER_1M_OUTPUT_TOKENS
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    start = time.time()
    cached = cache.lookup(req.prompt)

    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        cost_saved = _estimate_cost(
            cached["input_tokens_saved"], cached["output_tokens_saved"]
        )
        recent_requests.appendleft({
            "prompt": req.prompt,
            "status": "hit",
            "matched_query": cached["matched_query"],
            "similarity": round(cached["similarity"], 4),
            "latency_ms": round(latency_ms, 1),
            "cost_saved": round(cost_saved, 6),
            "timestamp": time.time(),
        })
        return ChatResponse(
            response=cached["response"],
            cache_status="hit",
            latency_ms=latency_ms,
            similarity=cached["similarity"],
        )

    # Cache miss -> actually call the LLM
    try:
        result = call_llm(req.prompt)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))

    cache.store(req.prompt, result["text"])

    recent_requests.appendleft({
        "prompt": req.prompt,
        "status": "miss",
        "matched_query": None,
        "similarity": None,
        "latency_ms": round(result["latency_ms"], 1),
        "cost_saved": 0.0,
        "timestamp": time.time(),
    })

    return ChatResponse(
        response=result["text"],
        cache_status="miss",
        latency_ms=result["latency_ms"],
        similarity=None,
    )


@app.get("/stats")
def stats():
    base = cache.stats()
    total_requests = len(recent_requests)
    hits = sum(1 for r in recent_requests if r["status"] == "hit")
    total_cost_saved = sum(r["cost_saved"] for r in recent_requests)

    hit_latencies = [r["latency_ms"] for r in recent_requests if r["status"] == "hit"]
    miss_latencies = [r["latency_ms"] for r in recent_requests if r["status"] == "miss"]

    return {
        **base,
        "requests_in_window": total_requests,
        "hit_rate_pct": round((hits / total_requests) * 100, 1) if total_requests else 0.0,
        "total_cost_saved_usd": round(total_cost_saved, 6),
        "avg_hit_latency_ms": round(sum(hit_latencies) / len(hit_latencies), 1) if hit_latencies else None,
        "avg_miss_latency_ms": round(sum(miss_latencies) / len(miss_latencies), 1) if miss_latencies else None,
        "similarity_threshold": config.SIMILARITY_THRESHOLD,
        "llm_provider": config.LLM_PROVIDER,
    }


@app.get("/recent")
def recent(limit: int = 20):
    return list(recent_requests)[:limit]


@app.get("/health")
def health():
    return {"status": "ok"}
