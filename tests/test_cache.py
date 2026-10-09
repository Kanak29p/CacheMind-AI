"""
test_cache.py

Unit tests for SemanticCache engine:
1. Cache hit & miss
2. TTL Expiry
3. Similarity Threshold boundaries
4. Multi-tenant Namespace Isolation
"""

import os
import sys
import time
import pytest

# Ensure backend directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import config
from cache import SemanticCache


@pytest.fixture
def temp_cache(tmp_path):
    """Fixture providing isolated temporary SemanticCache instance."""
    db_file = str(tmp_path / "test_cache.sqlite")
    index_file = str(tmp_path / "test_faiss.bin")

    # Override config paths for testing
    config.CACHE_DB_PATH = db_file
    config.CACHE_INDEX_PATH = index_file
    config.SIMILARITY_THRESHOLD = 0.80

    sc = SemanticCache()
    yield sc


def test_cache_hit_and_miss(temp_cache):
    """Test storing a query and verifying exact & paraphrased hit vs distinct miss."""
    query = "How do I reset my account password?"
    response = "Go to settings and click reset password."

    # Store entry
    temp_cache.store(query, response, model="llama3", temperature=0.7, namespace="user_a")

    # 1. Exact query lookup -> HIT
    hit = temp_cache.lookup(query, model="llama3", temperature=0.7, namespace="user_a")
    assert hit is not None
    assert hit["response"] == response
    assert hit["matched_query"] == query
    assert hit["similarity"] >= 0.99

    # 2. Paraphrased query lookup -> HIT
    para = "How do I reset my forgotten password?"
    hit_para = temp_cache.lookup(para, model="llama3", temperature=0.7, namespace="user_a")
    assert hit_para is not None
    assert hit_para["response"] == response

    # 3. Completely different query -> MISS
    miss = temp_cache.lookup("What is the refund policy for orders?", model="llama3", temperature=0.7, namespace="user_a")
    assert miss is None


def test_ttl_expiry(temp_cache, monkeypatch):
    """Test lazy eviction of expired entries based on CACHE_TTL_SECONDS."""
    query = "What are your business operating hours?"
    response = "We are open 9am-5pm EST."

    # Store entry with short TTL (override config TTL to 1 second)
    monkeypatch.setattr(config, "CACHE_TTL_SECONDS", 1)
    temp_cache.store(query, response)

    # Immediate lookup -> HIT
    assert temp_cache.lookup(query) is not None

    # Wait for TTL to expire
    time.sleep(1.2)

    # Lookup after TTL expiry -> MISS (and lazy deleted)
    expired = temp_cache.lookup(query)
    assert expired is None
    assert len(temp_cache.metadata) == 0


def test_threshold_boundary(temp_cache, monkeypatch):
    """Test that queries below SIMILARITY_THRESHOLD are rejected as misses."""
    # Set strict similarity threshold
    monkeypatch.setattr(config, "SIMILARITY_THRESHOLD", 0.98)

    query = "How do I cancel my subscription?"
    response = "Navigate to billing tab and press Cancel."

    temp_cache.store(query, response)

    # Query with slightly different wording (similarity < 0.98) -> MISS
    near_miss = temp_cache.lookup("How do I pause my subscription?", model="default")
    assert near_miss is None


def test_namespace_isolation(temp_cache):
    """Test multi-tenant scoping by model, system_prompt, temperature, and namespace."""
    query = "Who are you?"
    resp_pirate = "Ahoy! I be a pirate bot."
    resp_formal = "Greetings. I am a formal corporate assistant."

    # Store pirate persona under system_prompt "pirate" and namespace "ns_pirate"
    temp_cache.store(query, resp_pirate, model="llama3", system_prompt="Speak like a pirate", temperature=0.7, namespace="ns_pirate")

    # Store formal persona under system_prompt "formal" and namespace "ns_formal"
    temp_cache.store(query, resp_formal, model="llama3", system_prompt="Speak formally", temperature=0.7, namespace="ns_formal")

    # 1. Lookup under pirate namespace -> retrieves pirate response
    res_p = temp_cache.lookup(query, model="llama3", system_prompt="Speak like a pirate", temperature=0.7, namespace="ns_pirate")
    assert res_p is not None
    assert res_p["response"] == resp_pirate

    # 2. Lookup under formal namespace -> retrieves formal response
    res_f = temp_cache.lookup(query, model="llama3", system_prompt="Speak formally", temperature=0.7, namespace="ns_formal")
    assert res_f is not None
    assert res_f["response"] == resp_formal

    # 3. Cross-tenant lookup (pirate query with formal namespace) -> MISS
    cross = temp_cache.lookup(query, model="llama3", system_prompt="Speak like a pirate", temperature=0.7, namespace="ns_formal")
    assert cross is None
