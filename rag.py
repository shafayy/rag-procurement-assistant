"""
rag.py — Core RAG engine (Google Gemini edition — 100% free tier).

Stack
-----
  Generation : Gemini 2.0 Flash  (google-generativeai)
  Embeddings : models/text-embedding-004  (same library, free)
  Vector DB  : FAISS if installed, pure-Python cosine fallback otherwise

Flow
----
  1. Load documents from /docs  (.txt, .md, .pdf)
  2. Chunk with overlap
  3. Embed every chunk via Gemini Embeddings API
  4. Store in FAISS / in-memory index
  5. Query → embed → nearest-neighbour → stuff context → Gemini Flash → stream answer
"""

import os
import json
import math
import re
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import google.generativeai as genai

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GENERATION_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL  = "models/text-embedding-004"

try:
    import faiss
    import numpy as np
    USE_FAISS = True
except ImportError:
    USE_FAISS = False


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_vector(text: str, vocab: list[str]) -> list[float]:
    tokens = re.findall(r"\w+", text.lower())
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    vec = [float(freq.get(w, 0)) for w in vocab]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not GEMINI_API_KEY:
        vocab = list({w for t in texts for w in re.findall(r"\w+", t.lower())})
        return [_tfidf_vector(t, vocab) for t in texts]
    results = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type="retrieval_document",
        )
        results.extend(resp["embedding"])
    return results


def embed_query(query: str, vocab: Optional[list[str]] = None) -> list[float]:
    if not GEMINI_API_KEY:
        return _tfidf_vector(query, vocab or [])
    resp = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return resp["embedding"]


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.chunks:  list[str]         = []
        self.sources: list[str]         = []
        self._matrix: list[list[float]] = []
        self._vocab:  list[str]         = []
        self._index                     = None

    def build(self, chunks: list[str], sources: list[str]):
        self.chunks  = chunks
        self.sources = sources
        print(f"  Embedding {len(chunks)} chunks ...")
        vecs = embed_texts(chunks)
        if not GEMINI_API_KEY:
            self._vocab = list({w for c in chunks for w in re.findall(r"\w+", c.lower())})
        if USE_FAISS:
            mat = np.array(vecs, dtype="float32")
            faiss.normalize_L2(mat)
            self._index = faiss.IndexFlatIP(mat.shape[1])
            self._index.add(mat)
        else:
            self._matrix = vecs

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks:
            return []
        qvec = embed_query(query, self._vocab)
        if USE_FAISS:
            q = np.array([qvec], dtype="float32")
            faiss.normalize_L2(q)
            scores, idxs = self._index.search(q, min(top_k, len(self.chunks)))
            return [
                {"text": self.chunks[i], "source": self.sources[i], "score": float(scores[0][r])}
                for r, i in enumerate(idxs[0]) if i != -1
            ]
        else:
            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na  = math.sqrt(sum(x * x for x in a)) or 1
                nb  = math.sqrt(sum(x * x for x in b)) or 1
                return dot / (na * nb)
            scored = sorted(range(len(self._matrix)), key=lambda i: cosine(qvec, self._matrix[i]), reverse=True)
            return [
                {"text": self.chunks[i], "source": self.sources[i], "score": cosine(qvec, self._matrix[i])}
                for i in scored[:top_k]
            ]

    def save(self, path: str = "index.json"):
        with open(path, "w") as f:
            json.dump({"chunks": self.chunks, "sources": self.sources,
                       "matrix": self._matrix, "vocab": self._vocab}, f)
        print(f"  Index saved -> {path}")

    def load(self, path: str = "index.json"):
        with open(path) as f:
            d = json.load(f)
        self.chunks  = d["chunks"]
        self.sources = d["sources"]
        self._matrix = d["matrix"]
        self._vocab  = d["vocab"]
        if USE_FAISS and self._matrix:
            mat = np.array(self._matrix, dtype="float32")
            faiss.normalize_L2(mat)
            self._index = faiss.IndexFlatIP(mat.shape[1])
            self._index.add(mat)
        print(f"  Index loaded <- {path}  ({len(self.chunks)} chunks)")


# ─────────────────────────────────────────────────────────────────────────────
# Document loader
# ─────────────────────────────────────────────────────────────────────────────

def load_docs(folder: str = "docs") -> list[tuple[str, str]]:
    results = []
    for p in Path(folder).rglob("*"):
        if p.suffix in {".txt", ".md"}:
            results.append((p.name, p.read_text(errors="ignore")))
        elif p.suffix == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(p) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                results.append((p.name, text))
            except Exception as e:
                print(f"  [warn] Could not read {p.name}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RAG Assistant
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable procurement and supply-chain analyst assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Always cite the source document name when referencing specific information.

CONTEXT:
{context}"""


class RAGAssistant:
    """
    End-to-end RAG assistant powered by Google Gemini (free tier).

    Usage
    -----
    rag = RAGAssistant()
    rag.ingest("docs/")
    answer = rag.ask("What is the reorder policy for PPE?")
    """

    def __init__(self, docs_folder: str = "docs", index_path: str = "index.json"):
        self.store       = VectorStore()
        self.docs_folder = docs_folder
        self.index_path  = index_path
        self.history: list[dict] = []
        self._model = genai.GenerativeModel(GENERATION_MODEL) if GEMINI_API_KEY else None

    def ingest(self, folder: Optional[str] = None):
        folder = folder or self.docs_folder
        docs   = load_docs(folder)
        if not docs:
            print(f"  [warn] No documents found in '{folder}'. Add .txt / .md / .pdf files.")
            return
        all_chunks, all_sources = [], []
        for name, text in docs:
            c = chunk_text(text)
            all_chunks.extend(c)
            all_sources.extend([name] * len(c))
            print(f"  Loaded: {name}  ({len(c)} chunks)")
        self.store.build(all_chunks, all_sources)
        self.store.save(self.index_path)

    def load_index(self):
        self.store.load(self.index_path)

    def ask(self, question: str, top_k: int = 5, stream: bool = True) -> str:
        results = self.store.search(question, top_k=top_k)
        context = (
            "\n\n---\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
            if results else "No relevant documents found in the knowledge base."
        )
        system_with_context = SYSTEM_PROMPT.format(context=context)

        if not self.history:
            gemini_history = [
                {"role": "user",  "parts": [system_with_context]},
                {"role": "model", "parts": ["Understood. I will answer strictly from the provided context and cite sources."]},
            ]
        else:
            gemini_history = self.history.copy()
            gemini_history[0] = {"role": "user", "parts": [system_with_context]}

        gemini_history.append({"role": "user", "parts": [question]})

        if not self._model:
            print("[error] GEMINI_API_KEY not set. Add it to your .env file.")
            return ""

        if stream:
            print("\nAssistant: ", end="", flush=True)
            full = ""
            response = self._model.generate_content(gemini_history, stream=True)
            for chunk in response:
                text = chunk.text or ""
                print(text, end="", flush=True)
                full += text
            print()
        else:
            response = self._model.generate_content(gemini_history)
            full = response.text
            print(f"\nAssistant: {full}")

        if not self.history:
            self.history = [
                {"role": "user",  "parts": [system_with_context]},
                {"role": "model", "parts": ["Understood. I will answer strictly from the provided context and cite sources."]},
            ]
        self.history.append({"role": "user",  "parts": [question]})
        self.history.append({"role": "model", "parts": [full]})
        return full

    def reset_history(self):
        self.history = []
