# RAG Procurement Assistant

A lightweight, production-ready Retrieval-Augmented Generation (RAG) system for answering questions over procurement and supply chain documents — powered entirely by **Google Gemini** (free tier, no credit card required).

Built by Shafay Iqbal as a practical demonstration of RAG architecture applied to enterprise procurement knowledge management.

---

## Stack (100% free)

| Layer         | Technology                                      |
| ------------- | ----------------------------------------------- |
| Generation    | Gemini 2.0 Flash (`google-generativeai`)        |
| Embeddings    | Gemini `text-embedding-004`                     |
| Vector search | FAISS (optional) or pure-Python cosine fallback |
| PDF support   | pdfplumber                                      |

---

## Architecture

```
User Question
     │
     ▼
Embed Query (Gemini text-embedding-004)
     │
     ▼
FAISS / Cosine Search over document chunks
     │
     ▼
Top-K Chunks injected as context
     │
     ▼
Gemini 2.0 Flash (streaming generation)
     │
     ▼
Answer (with source citations)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/shafayiqbal/rag-procurement-assistant.git
cd rag-procurement-assistant
pip install -r requirements.txt
```

### 2. Get your free API key

Go to **https://aistudio.google.com/apikey** — takes 30 seconds, no credit card.

```bash
cp .env.example .env
# Add your key:  GEMINI_API_KEY=your_key_here
```

### 3. Run the demo (no documents needed)

```bash
python main.py demo
```

Injects three synthetic procurement docs and runs four questions end-to-end immediately.

### 4. Use your own documents

```bash
# Drop .txt / .md / .pdf files into /docs, then:
python main.py ingest      # build and save the index
python main.py chat        # interactive multi-turn session
python main.py ask "What is the reorder policy for PPE?"
```

---

## Running tests

```bash
pytest tests/ -v
```

12 unit tests — all pass with no API calls required.

Note: The code will prefer the top-level `genai` package if installed, and
fall back to `google.generativeai`. Install either package; `requirements.txt`
includes `google-generativeai`.

---

## Project structure

```
rag-procurement-assistant/
├── rag.py              # Core engine: chunking, embeddings, vector store, RAG pipeline
├── main.py             # CLI: ingest / chat / ask / demo
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/               # Drop your documents here
└── tests/
    └── test_rag.py     # 11 unit tests, no API required
```

---

## Why this exists

Demonstrates applied RAG in a real-world context: procurement knowledge management across a multi-entity healthcare network (Rafed UAE / PureHealth Group). The same architecture pattern underpins on-premise RAG chatbots evaluated for vendor affairs query handling across the SEHA hospital network.

---

## License

MIT
