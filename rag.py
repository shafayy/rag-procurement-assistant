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
from typing import Optional, List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

genai = None
try:
    import google.genai as _genai
    genai = _genai
except Exception:
    try:
        import genai as _genai
        genai = _genai
    except Exception:
        try:
            import google.generativeai as _genai
            genai = _genai
        except Exception:
            genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        # Ignore configuration errors during import-time; runtime will fail clearly if needed
        pass

# Flag used by tests to force fallback embeddings when True/False is toggled.
# Default to True only if we have both the library and an API key.
USE_OPENAI_EMBEDDINGS = bool(genai is not None and GEMINI_API_KEY)

# Allow forcing an offline/demo-only mode via env var. When enabled, no remote
# embedding or generation calls will be attempted.
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "").lower() in {"1", "true", "yes"}

# If a genai client can be constructed, prefer it for embeddings/generation.
genai_client = None
if genai is not None and GEMINI_API_KEY and not OFFLINE_MODE:
    try:
        # google.genai exposes Client; other packages may too
        if hasattr(genai, 'Client'):
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
        elif hasattr(genai, 'GenerativeModel'):
            genai_client = None
        else:
            genai_client = None
    except Exception:
        genai_client = None

# If we have a genai_client, consider embeddings enabled
if genai_client is not None:
    USE_OPENAI_EMBEDDINGS = True

# Allow forcing an offline/demo-only mode via env var. When enabled, no remote
# embedding or generation calls will be attempted.
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "").lower() in {"1", "true", "yes"}

# Sensible defaults; will be used when the client is available. These are more
# likely to be supported by modern `google-genai`.
GENERATION_MODEL = os.getenv('GENERATION_MODEL', "models/gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', "models/gemini-embedding-001")

try:
    import faiss
    import numpy as np
    USE_FAISS = True
except ImportError:
    USE_FAISS = False


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
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

def _tfidf_vector(text: str, vocab: List[str]) -> List[float]:
    tokens = re.findall(r"\w+", text.lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    vec = [float(freq.get(w, 0)) for w in vocab]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_texts(texts: List[str]) -> List[List[float]]:
    # If embeddings are not available or configured, use TF-IDF fallback.
    if not USE_OPENAI_EMBEDDINGS or genai is None:
        vocab = list({w for t in texts for w in re.findall(r"\w+", t.lower())})
        embed_texts._last_vocab = vocab
        return [_tfidf_vector(t, vocab) for t in texts]

    # Try the remote embeddings, but gracefully fall back to TF-IDF on failure.
    try:
        # If we have a genai client (google-genai), use its embed API
        if genai_client is not None:
            results = []
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = genai_client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
                # resp.embeddings -> list of objects with .values
                if hasattr(resp, 'embeddings'):
                    for e in resp.embeddings:
                        vals = getattr(e, 'values', None)
                        if vals is None and hasattr(e, 'embedding'):
                            vals = getattr(e.embedding, 'values', None)
                        results.append(list(vals) if vals is not None else [])
                else:
                    raise RuntimeError('Unexpected embed response format')
            embed_texts._last_vocab = []
            return results
        # Fallback to older genai/google.generativeai API shape
        results = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document",
            )
            if isinstance(resp, dict) and "embedding" in resp:
                results.extend(resp["embedding"])
            elif hasattr(resp, "embedding"):
                results.extend(resp.embedding)
            else:
                raise RuntimeError("Unexpected embedding response format")
        embed_texts._last_vocab = []
        return results
    except Exception as e:
        print(f"  [warn] Embeddings API failed, falling back to TF-IDF: {e}")
        vocab = list({w for t in texts for w in re.findall(r"\w+", t.lower())})
        embed_texts._last_vocab = vocab
        return [_tfidf_vector(t, vocab) for t in texts]


def embed_query(query: str, vocab: Optional[List[str]] = None) -> List[float]:
    if not USE_OPENAI_EMBEDDINGS or genai is None:
        return _tfidf_vector(query, vocab or [])

    try:
        if genai_client is not None:
            resp = genai_client.models.embed_content(model=EMBEDDING_MODEL, contents=[query])
            if hasattr(resp, 'embeddings') and resp.embeddings:
                vals = getattr(resp.embeddings[0], 'values', None)
                if vals is not None:
                    return list(vals)
        else:
            resp = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query",
            )
            if isinstance(resp, dict) and "embedding" in resp:
                return resp["embedding"]
            if hasattr(resp, "embedding"):
                return resp.embedding
        raise RuntimeError("Unexpected embedding response format")
    except Exception as e:
        print(f"  [warn] Embeddings API failed for query, falling back to TF-IDF: {e}")
        return _tfidf_vector(query, vocab or [])


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self._matrix: List[List[float]] = []
        self._vocab: List[str] = []
        self._index                     = None

    def build(self, chunks: List[str], sources: List[str]):
        self.chunks  = chunks
        self.sources = sources
        print(f"  Embedding {len(chunks)} chunks ...")
        vecs = embed_texts(chunks)
        # If embed_texts populated a local TF-IDF vocab, use it. Otherwise
        # fall back to deriving vocab from chunks when API is not used.
        last_vocab = getattr(embed_texts, "_last_vocab", None)
        if last_vocab:
            self._vocab = last_vocab
        elif not USE_OPENAI_EMBEDDINGS or genai is None:
            self._vocab = list({w for c in chunks for w in re.findall(r"\w+", c.lower())})
        if USE_FAISS:
            mat = np.array(vecs, dtype="float32")
            faiss.normalize_L2(mat)
            self._index = faiss.IndexFlatIP(mat.shape[1])
            self._index.add(mat)
        else:
            self._matrix = vecs

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
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

def load_docs(folder: str = "docs") -> List[tuple]:
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
        # Remote client (google-genai) or legacy model wrapper (google.generativeai)
        self._client = genai_client
        self._model = None
        if genai is not None and GEMINI_API_KEY and not OFFLINE_MODE:
            # Legacy API may expose GenerativeModel
            if hasattr(genai, 'GenerativeModel'):
                try:
                    self._model = genai.GenerativeModel(GENERATION_MODEL)
                except Exception:
                    self._model = None

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

        # If offline mode is enabled or no remote model/client is configured,
        # perform a simple extractive answer based on the retrieved context
        # instead of calling the generation API.
        if OFFLINE_MODE or (self._client is None and self._model is None):
            # Build a concise extractive answer by selecting sentences that
            # contain query keywords from the top results.
            q_tokens = set(re.findall(r"\w+", question.lower()))
            extractive_sentences = []
            for r in results:
                text = r["text"]
                # split into sentences
                sents = re.split(r'(?<=[.!?])\s+', text)
                for s in sents:
                    st = s.strip()
                    if not st:
                        continue
                    toks = set(re.findall(r"\w+", st.lower()))
                    if q_tokens & toks:
                        extractive_sentences.append((st, r["source"]))
                if len(extractive_sentences) >= 3:
                    break

            if not extractive_sentences:
                # fallback: use the top-k chunks truncated
                for r in results[:3]:
                    extractive_sentences.append((r["text"][:300].strip(), r["source"]))

            answer_parts = [f"{s} (Source: {src})" for s, src in extractive_sentences]
            full = "\n\n".join(answer_parts) if answer_parts else "No relevant information found in the documents."
            print("\nAssistant (extractive offline):\n", full)
            # store into history and return
            if not self.history:
                self.history = [
                    {"role": "user",  "parts": [system_with_context]},
                    {"role": "model", "parts": ["Understood. I will answer strictly from the provided context and cite sources."]},
                ]
            self.history.append({"role": "user",  "parts": [question]})
            self.history.append({"role": "model", "parts": [full]})
            return full

        full = ""
        try:
            # Prefer new genai client streaming API when available
            if genai_client is not None:
                if stream and hasattr(genai_client.models, 'generate_content_stream'):
                    print("\nAssistant: ", end="", flush=True)
                    stream_iter = genai_client.models.generate_content_stream(model=GENERATION_MODEL, contents=[system_with_context, question])
                    for resp in stream_iter:
                        # resp may have .candidates with Content objects
                        text = ''
                        if hasattr(resp, 'candidates') and resp.candidates:
                            cand = resp.candidates[0]
                            cont = getattr(cand, 'content', None)
                            if cont and hasattr(cont, 'parts'):
                                parts = []
                                for p in cont.parts:
                                    parts.append(getattr(p, 'text', '') or '')
                                text = ''.join(parts)
                            else:
                                text = getattr(cand, 'text', '') or ''
                        elif hasattr(resp, 'output'):
                            text = str(resp.output)
                        print(text, end="", flush=True)
                        full += text
                    print()
                else:
                    resp = genai_client.models.generate_content(model=GENERATION_MODEL, contents=[system_with_context, question])
                    # Try typical response shapes
                    if hasattr(resp, 'candidates') and resp.candidates:
                        cand = resp.candidates[0]
                        cont = getattr(cand, 'content', None)
                        if cont and hasattr(cont, 'parts'):
                            parts = [getattr(p, 'text', '') or '' for p in cont.parts]
                            full = ''.join(parts)
                        else:
                            full = getattr(cand, 'text', None) or getattr(cand, 'output', None) or str(cand)
                    else:
                        full = str(resp)
                    print(f"\nAssistant: {full}")
            else:
                # Old-style genai/generativeai model
                if stream:
                    print("\nAssistant: ", end="", flush=True)
                    response = self._model.generate_content(gemini_history, stream=True)
                    for chunk in response:
                        text = chunk.text or ""
                        print(text, end="", flush=True)
                        full += text
                    print()
                else:
                    response = self._model.generate_content(gemini_history)
                    full = getattr(response, "text", str(response))
                    print(f"\nAssistant: {full}")
        except Exception as e:
            # Graceful fallback when the generation API is unavailable (quota, model mismatch, etc.)
            print(f"\n  [warn] Generation API failed: {e}")
            # Provide an extractive fallback: show the context snippets as the assistant reply.
            fallback = context if context else "No context available to form an answer."
            print("\nAssistant (fallback):\n", fallback)
            full = fallback

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
