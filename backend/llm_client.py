"""
Thin client for whichever LLM actually answers cache-miss queries.

Engineering & Design Decisions:
1. Async Non-Blocking HTTP Client (httpx.AsyncClient):
   - Replaced blocking requests with httpx.AsyncClient so FastAPI async endpoints
     do not block event loop processing during external LLM API calls.
2. Real Token Extraction (usage field):
   - Extracts real token counts directly from API response `usage` metadata:
     - Groq: `prompt_tokens` & `completion_tokens`
     - Ollama: `prompt_eval_count` & `eval_count`
   - Falls back to `len(text) // 4` only when usage metadata is absent.
"""

import asyncio
import logging
import time
import httpx
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


async def call_llm_async(
    prompt: str,
    model: str = "default",
    system_prompt: str = "",
    temperature: float = 0.7,
) -> dict:
    """Returns {"text": str, "latency_ms": float, "prompt_tokens": int, "completion_tokens": int}"""
    start = time.time()

    if config.LLM_PROVIDER == "groq":
        if not config.GROQ_API_KEY:
            await asyncio.sleep(0.15)  # Simulate network latency
            text = f"[Mock LLM Response] Answer for: '{prompt}'. (Add GROQ_API_KEY in backend/.env for live Groq AI responses)"
            res = {
                "text": text,
                "prompt_tokens": max(1, len(prompt) // config.CHARS_PER_TOKEN),
                "completion_tokens": max(1, len(text) // config.CHARS_PER_TOKEN),
            }
        else:
            res = await _call_groq_async(prompt, model=model, system_prompt=system_prompt, temperature=temperature)
    elif config.LLM_PROVIDER == "ollama":
        res = await _call_ollama_async(prompt, model=model, system_prompt=system_prompt, temperature=temperature)
    elif config.LLM_PROVIDER == "mock":
        await asyncio.sleep(0.15)
        text = f"[Mock LLM Response] Answer for: '{prompt}'"
        res = {
            "text": text,
            "prompt_tokens": max(1, len(prompt) // config.CHARS_PER_TOKEN),
            "completion_tokens": max(1, len(text) // config.CHARS_PER_TOKEN),
        }
    else:
        raise LLMError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")

    latency_ms = (time.time() - start) * 1000
    res["latency_ms"] = latency_ms
    return res


def call_llm(
    prompt: str,
    model: str = "default",
    system_prompt: str = "",
    temperature: float = 0.7,
) -> dict:
    """Synchronous fallback wrapper."""
    return asyncio.run(call_llm_async(prompt, model, system_prompt, temperature))


async def _call_groq_async(
    prompt: str,
    model: str = "default",
    system_prompt: str = "",
    temperature: float = 0.7,
) -> dict:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY not set in environment.")

    target_model = config.GROQ_MODEL if model == "default" else model
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                config.GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

            # Extract real token counts from usage metadata
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", max(1, len(prompt) // config.CHARS_PER_TOKEN))
            completion_tokens = usage.get("completion_tokens", max(1, len(text) // config.CHARS_PER_TOKEN))

            return {
                "text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        except Exception as e:
            raise LLMError(f"Groq API call failed: {str(e)}")


async def _call_ollama_async(
    prompt: str,
    model: str = "default",
    system_prompt: str = "",
    temperature: float = 0.7,
) -> dict:
    target_model = config.OLLAMA_MODEL if model == "default" else model
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                config.OLLAMA_API_URL,
                json={
                    "model": target_model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")

            # Extract real token counts from Ollama response metadata
            prompt_tokens = data.get("prompt_eval_count", max(1, len(prompt) // config.CHARS_PER_TOKEN))
            completion_tokens = data.get("eval_count", max(1, len(text) // config.CHARS_PER_TOKEN))

            return {
                "text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        except Exception as e:
            raise LLMError(f"Ollama API call failed: {str(e)}")
