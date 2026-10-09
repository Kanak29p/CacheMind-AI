"""
CacheMind AI Proxy Server (FastAPI)

Features & API Routes:
1. POST /chat                      -> Custom proxy endpoint
2. POST /v1/chat/completions       -> Drop-in OpenAI API compatible endpoint
3. GET  /stats                     -> Real-time metric aggregate for dashboard
4. GET  /eval                      -> Evaluation threshold sweep metrics & optimal threshold
5. GET  /recent                    -> Recent request audit feed
6. GET  /health                    -> Health check & system status
7. GET  /                          -> Dashboard UI (Hugging Face Spaces root landing)

Production Middleware & Security:
- Optional API Key Authentication (X-API-Key header or Authorization Bearer)
- Sliding Window IP Rate Limiter
- Non-blocking async execution (FAISS + SQLite dispatched to threadpool via run_in_threadpool)
"""

import json
import os
import time
from collections import deque, defaultdict
from typing import Literal, Optional, List

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from cache import SemanticCache
from llm_client import call_llm_async, validate_provider_config, LLMError

START_TIME = time.time()

app = FastAPI(
    title="CacheMind AI - Semantic Caching LLM Proxy",
    description="Drop-in multi-tenant semantic caching proxy for LLMs",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core semantic cache engine
cache = SemanticCache()

# In-memory request feed buffer for dashboard monitoring (max 200 items)
recent_requests = deque(maxlen=200)

# In-memory sliding window rate limiter state: IP -> list of timestamps
rate_limit_history = defaultdict(list)


# ---------- Security & Auth Dependencies ----------
def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """Optional API Key validation via environment variable API_KEY."""
    if not config.API_KEY:
        return True  # Auth disabled if API_KEY env variable is not set

    token = x_api_key
    if not token and authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()

    if token != config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key.",
        )
    return True


def check_rate_limit(request: Request):
    """Basic sliding-window rate limiter per client IP address."""
    if config.RATE_LIMIT_PER_MINUTE <= 0:
        return

    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()
    window_start = now - 60.0

    # Filter out timestamps older than 60 seconds
    timestamps = [t for t in rate_limit_history[client_ip] if t > window_start]
    if len(timestamps) >= config.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Too Many Requests: Rate limit of {config.RATE_LIMIT_PER_MINUTE} requests/min exceeded.",
        )

    timestamps.append(now)
    rate_limit_history[client_ip] = timestamps


@app.on_event("startup")
def startup_event():
    validate_provider_config()


# ---------- Request / Response Schemas ----------
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User query or prompt text")
    model: str = Field("default", description="Target LLM model name")
    system_prompt: str = Field("", description="System instructions or persona")
    temperature: float = Field(0.7, description="Sampling temperature")
    namespace: str = Field("default", description="Multi-tenant user/organization namespace")


class ChatResponse(BaseModel):
    response: str
    cache_status: Literal["hit", "miss"]
    latency_ms: float
    similarity: Optional[float] = None
    matched_prompt: Optional[str] = None


# OpenAI Chat Completion Specification Schemas
class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "default"
    messages: List[OpenAIMessage]
    temperature: float = 0.7
    user: Optional[str] = None
    namespace: Optional[str] = "default"


# ---------- Utilities ----------
def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * config.COST_PER_1M_INPUT_TOKENS
        + (output_tokens / 1_000_000) * config.COST_PER_1M_OUTPUT_TOKENS
    )


# ---------- API Routes ----------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves Dashboard HTML UI at root '/' for local usage & Hugging Face Spaces deployment."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    if not os.path.exists(dashboard_path):
        dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")

    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>CacheMind AI Proxy is Running!</h2><p>Dashboard file index.html not found.</p>")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def chat(req: ChatRequest):
    """Custom Chat Proxy Endpoint with Semantic Cache Lookup."""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    start = time.time()
    cached = await run_in_threadpool(
        cache.lookup,
        req.prompt,
        model=req.model,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        namespace=req.namespace,
    )

    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        cost_saved = _estimate_cost(cached["input_tokens_saved"], cached["output_tokens_saved"])
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
            matched_prompt=cached["matched_query"],
        )

    # Cache miss -> async non-blocking LLM call
    try:
        result = await call_llm_async(
            req.prompt,
            model=req.model,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))

    await run_in_threadpool(
        cache.store,
        req.prompt,
        result["text"],
        model=req.model,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        namespace=req.namespace,
        input_tokens=result.get("prompt_tokens"),
        output_tokens=result.get("completion_tokens"),
    )

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
        matched_prompt=None,
    )


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def openai_chat_completions(req: OpenAIChatRequest):
    """Drop-in OpenAI Chat Completions API compatibility layer."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")

    # Extract user prompt (last user role message) and system prompt
    prompt = ""
    system_prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            prompt = msg.content

    if not prompt:
        raise HTTPException(status_code=400, detail="No user message found in payload.")

    ns = req.namespace or req.user or "default"
    start = time.time()

    cached = await run_in_threadpool(
        cache.lookup,
        prompt,
        model=req.model,
        system_prompt=system_prompt,
        temperature=req.temperature,
        namespace=ns,
    )

    now_ts = int(time.time())
    completion_id = f"chatcmpl-cachemind-{now_ts}"

    if cached is not None:
        latency_ms = (time.time() - start) * 1000
        cost_saved = _estimate_cost(cached["input_tokens_saved"], cached["output_tokens_saved"])
        recent_requests.appendleft({
            "prompt": prompt,
            "status": "hit",
            "matched_query": cached["matched_query"],
            "similarity": round(cached["similarity"], 4),
            "latency_ms": round(latency_ms, 1),
            "cost_saved": round(cost_saved, 6),
            "timestamp": time.time(),
        })

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": now_ts,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": cached["response"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": cached["input_tokens_saved"],
                "completion_tokens": cached["output_tokens_saved"],
                "total_tokens": cached["input_tokens_saved"] + cached["output_tokens_saved"],
            },
            "cache_status": "hit",
            "similarity": cached["similarity"],
        }

    # Cache miss -> Call LLM
    try:
        result = await call_llm_async(
            prompt,
            model=req.model,
            system_prompt=system_prompt,
            temperature=req.temperature,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))

    await run_in_threadpool(
        cache.store,
        prompt,
        result["text"],
        model=req.model,
        system_prompt=system_prompt,
        temperature=req.temperature,
        namespace=ns,
        input_tokens=result.get("prompt_tokens"),
        output_tokens=result.get("completion_tokens"),
    )

    p_tokens = result.get("prompt_tokens", len(prompt) // config.CHARS_PER_TOKEN)
    c_tokens = result.get("completion_tokens", len(result["text"]) // config.CHARS_PER_TOKEN)

    recent_requests.appendleft({
        "prompt": prompt,
        "status": "miss",
        "matched_query": None,
        "similarity": None,
        "latency_ms": round(result["latency_ms"], 1),
        "cost_saved": 0.0,
        "timestamp": time.time(),
    })

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": now_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        },
        "cache_status": "miss",
    }


@app.get("/stats")
def stats():
    """Returns aggregated proxy metrics for real-time dashboard visualization."""
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


@app.get("/eval")
def get_eval_results():
    """Returns evaluation threshold sweep results & optimal threshold metrics."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.json"),
        os.path.join(os.path.dirname(__file__), "data", "eval_results.json"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    # Run eval threshold sweep if file does not exist yet
    from eval_threshold import run_threshold_sweep
    return run_threshold_sweep()


@app.get("/recent")
def recent(limit: int = 20):
    """Returns recent request history feed."""
    return list(recent_requests)[:limit]


@app.get("/health")
def health():
    """Health check endpoint for Docker & Hugging Face Spaces probes."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "llm_provider": config.LLM_PROVIDER,
        "active_cache_entries": len(cache.metadata),
    }
