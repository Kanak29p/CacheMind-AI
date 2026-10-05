"""
Thin client for whichever LLM actually answers cache-miss queries.

Two zero-cost options, switchable via config.LLM_PROVIDER:
- "groq": hosted, free-tier API (real dollar-cost math for your dashboard)
- "ollama": fully local model, zero cost, zero rate limits, offline demo
"""
import logging
import time
import requests
import config

logger = logging.getLogger("semantic_cache_proxy")


class LLMError(Exception):
    pass


def validate_provider_config():
    """Verify LLM provider setup on startup. Fail loudly if misconfigured."""
    logger.info(f"[STARTUP] Active LLM provider: '{config.LLM_PROVIDER}'")
    valid_providers = {"groq", "ollama", "mock"}
    if config.LLM_PROVIDER not in valid_providers:
        raise RuntimeError(
            f"Invalid LLM_PROVIDER '{config.LLM_PROVIDER}'. Must be one of {valid_providers}"
        )
    if config.LLM_PROVIDER == "groq" and not config.GROQ_API_KEY:
        logger.warning(
            "[STARTUP WARNING] GROQ_API_KEY not configured. Falling back to mock responses for cache miss queries."
        )


def call_llm(prompt: str) -> dict:
    """Returns {"text": str, "latency_ms": float}"""
    start = time.time()

    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY:
            text = f"[Mock LLM Response] Answer for: '{prompt}'. (Add GROQ_API_KEY in backend/.env for live Groq AI responses)"
            time.sleep(0.15)  # simulate network latency
        else:
            text = _call_groq(prompt)
    elif config.LLM_PROVIDER == "ollama":
        text = _call_ollama(prompt)
    elif config.LLM_PROVIDER == "mock":
        text = f"[Mock LLM Response] Answer for: '{prompt}'"
        time.sleep(0.15)
    else:
        raise LLMError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")

    latency_ms = (time.time() - start) * 1000
    return {"text": text, "latency_ms": latency_ms}


def _call_groq(prompt: str) -> str:
    if not config.GROQ_API_KEY:
        raise LLMError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "or switch LLM_PROVIDER=ollama in your .env for a fully local demo."
        )

    resp = requests.post(
        config.GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        config.OLLAMA_API_URL,
        json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")
