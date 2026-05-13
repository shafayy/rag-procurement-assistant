"""
tests/test_rag.py — Unit tests for the RAG engine (no API calls required).
Run with: pytest tests/ -v
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag import chunk_text, VectorStore, _tfidf_vector


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def test_chunk_basic():
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # every chunk should have at most 100 words
    for c in chunks:
        assert len(c.split()) <= 100


def test_chunk_overlap():
    text = " ".join([f"w{i}" for i in range(200)])
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    # consecutive chunks should share words (overlap)
    words_0 = set(chunks[0].split())
    words_1 = set(chunks[1].split())
    assert len(words_0 & words_1) > 0


def test_chunk_short_text():
    text = "hello world"
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_empty():
    chunks = chunk_text("", chunk_size=100, overlap=20)
    assert chunks == []


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF fallback vectors
# ─────────────────────────────────────────────────────────────────────────────

def test_tfidf_vector_unit_norm():
    vocab = ["procurement", "vendor", "invoice", "delivery"]
    v = _tfidf_vector("procurement vendor procurement", vocab)
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_tfidf_vector_zeros_for_unknown():
    vocab = ["apple", "banana"]
    v = _tfidf_vector("completely unrelated text", vocab)
    assert all(x == 0.0 for x in v)


def test_tfidf_vector_length_matches_vocab():
    vocab = ["a", "b", "c", "d", "e"]
    v = _tfidf_vector("a b c", vocab)
    assert len(v) == len(vocab)


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore (fallback mode, no OpenAI / FAISS)
# ─────────────────────────────────────────────────────────────────────────────

def _build_store():
    """Helper: build a small store with known content."""
    store = VectorStore()
    chunks = [
        "vendor certification ISO 13485 medical devices",
        "invoice payment terms net 30 days",
        "delivery schedule on time rate procurement",
        "item master UNSPSC classification healthcare SKU",
        "python automation pipeline data governance",
    ]
    sources = [f"doc{i}.txt" for i in range(len(chunks))]
    # Bypass OpenAI — patch embed_texts directly
    import rag
    original = rag.embed_texts
    rag.USE_OPENAI_EMBEDDINGS = False
    store.build(chunks, sources)
    rag.embed_texts = original
    return store, chunks


def test_store_build_populates():
    store, chunks = _build_store()
    assert len(store.chunks) == len(chunks)


def test_store_search_returns_results():
    store, _ = _build_store()
    import rag
    rag.USE_OPENAI_EMBEDDINGS = False
    results = store.search("vendor certification requirements", top_k=3)
    assert len(results) >= 1
    assert all("text" in r and "source" in r and "score" in r for r in results)


def test_store_search_top_k_respected():
    store, _ = _build_store()
    import rag
    rag.USE_OPENAI_EMBEDDINGS = False
    results = store.search("anything", top_k=2)
    assert len(results) <= 2


def test_store_search_empty_store():
    store = VectorStore()
    results = store.search("anything", top_k=5)
    assert results == []


def test_store_save_and_load(tmp_path):
    store, _ = _build_store()
    import rag
    rag.USE_OPENAI_EMBEDDINGS = False
    path = str(tmp_path / "index.json")
    store.save(path)

    store2 = VectorStore()
    store2.load(path)
    assert store2.chunks == store.chunks
    assert store2.sources == store.sources
