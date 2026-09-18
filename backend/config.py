"""
Central configuration for the semantic caching proxy.

Everything here is designed to run on free tiers / local compute:
- EMBEDDING_MODEL runs locally via sentence-transformers (no API cost)
- LLM_PROVIDER defaults to "groq" (generous free tier) with "ollama" as a
  fully offline, zero-cost fallback if you don't want to sign up for anything
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Embedding model (runs locally, free) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2

# --- Cache behavior ---
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.92"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60 * 24)))  # 24h default
CACHE_INDEX_PATH = os.getenv("CACHE_INDEX_PATH", "./data/faiss_index.bin")
CACHE_METADATA_PATH = os.getenv("CACHE_METADATA_PATH", "./data/cache_metadata.json")

# --- LLM provider ---
# "groq"   -> free-tier hosted API, real dollar-cost math, needs GROQ_API_KEY
# "ollama" -> fully local, zero cost, zero rate limits, needs Ollama running
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# --- Cost model (for the dashboard's "dollars saved" math) ---
# These are illustrative, editable rates ($ per 1M tokens). Update to match
# whatever model you're "simulating" savings against, e.g. GPT-4o-mini.
COST_PER_1M_INPUT_TOKENS = float(os.getenv("COST_PER_1M_INPUT_TOKENS", "0.15"))
COST_PER_1M_OUTPUT_TOKENS = float(os.getenv("COST_PER_1M_OUTPUT_TOKENS", "0.60"))
# Rough chars-per-token estimate for quick cost math without a real tokenizer
CHARS_PER_TOKEN = 4
