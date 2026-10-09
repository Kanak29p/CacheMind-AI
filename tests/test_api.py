"""
test_api.py

Integration tests for FastAPI endpoints:
1. GET /health
2. GET /stats
3. GET /eval
4. POST /chat (hit & miss)
5. POST /v1/chat/completions (OpenAI drop-in proxy compatibility)
"""

import os
import sys
import pytest
import httpx

# Ensure backend directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app
import config


@pytest.fixture
def override_test_config(tmp_path):
    """Fixture providing isolated temporary DB for API integration tests."""
    config.CACHE_DB_PATH = str(tmp_path / "api_test_cache.sqlite")
    config.CACHE_INDEX_PATH = str(tmp_path / "api_test_faiss.bin")
    config.LLM_PROVIDER = "mock"


@pytest.mark.asyncio
async def test_health_endpoint(override_test_config):
    """Verify GET /health returns 200 OK and system status."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_stats_and_eval_endpoints(override_test_config):
    """Verify GET /stats and GET /eval return metrics structures."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        stats_resp = await client.get("/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        assert "hit_rate_pct" in stats_data
        assert "total_cost_saved_usd" in stats_data

        eval_resp = await client.get("/eval")
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert "best_threshold" in eval_data
        assert "metrics_by_threshold" in eval_data


@pytest.mark.asyncio
async def test_chat_proxy_flow(override_test_config):
    """Verify POST /chat endpoint for cache miss followed by cache hit."""
    payload = {
        "prompt": "How do I update my profile picture?",
        "model": "llama3",
        "system_prompt": "Helpful bot",
        "temperature": 0.7,
        "namespace": "test_ns",
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. First request -> MISS
        res1 = await client.post("/chat", json=payload)
        assert res1.status_code == 200
        d1 = res1.json()
        assert d1["cache_status"] == "miss"
        assert d1["similarity"] is None

        # 2. Second request (exact prompt) -> HIT
        res2 = await client.post("/chat", json=payload)
        assert res2.status_code == 200
        d2 = res2.json()
        assert d2["cache_status"] == "hit"
        assert d2["similarity"] >= 0.99
        assert d2["matched_prompt"] == payload["prompt"]


@pytest.mark.asyncio
async def test_openai_chat_completions_flow(override_test_config):
    """Verify POST /v1/chat/completions OpenAI drop-in proxy compatibility."""
    openai_payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a customer service representative."},
            {"role": "user", "content": "Can I get a refund for my order?"},
        ],
        "temperature": 0.7,
        "user": "user_404",
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. First call -> MISS
        r1 = await client.post("/v1/chat/completions", json=openai_payload)
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["object"] == "chat.completion"
        assert body1["cache_status"] == "miss"
        assert len(body1["choices"]) == 1
        assert "usage" in body1

        # 2. Second call -> HIT
        r2 = await client.post("/v1/chat/completions", json=openai_payload)
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["object"] == "chat.completion"
        assert body2["cache_status"] == "hit"
        assert body2["usage"]["total_tokens"] > 0
