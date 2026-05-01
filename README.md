# chromadb-rag

A minimal Retrieval-Augmented Generation (RAG) system built with ChromaDB and Google Gemini. It ingests `.txt` documents, stores their embeddings in a local vector database, and answers natural-language questions grounded in those documents.

## How it works

```
docs/*.txt  →  chunk  →  embed (Gemini)  →  ChromaDB (local)
                                                    ↓
question  →  embed  →  similarity search  →  context  →  Gemini LLM  →  answer
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier available)

## Installation

```bash
git clone https://github.com/your-username/chromadb-rag.git
cd chromadb-rag

uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

The rest of the configuration lives in `src/config.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model used to embed documents and queries |
| `CHAT_MODEL` | `gemini-2.5-flash` | Model used to generate answers |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between consecutive chunks |
| `TOP_K` | `3` | Number of chunks retrieved per query |

## Usage

### 1. Add your documents

Place `.txt` files inside the `docs/` folder. The system ships with three example files:

```
docs/
  rag.txt
  chromadb.txt
  aws.txt
```

### 2. Ingest documents

Reads every `.txt` file in `docs/`, chunks the text, generates embeddings via Gemini, and upserts them into the local ChromaDB collection.

```bash
uv run python -m src.ingest
```

Output:

```
Files found: 3
Chunks generated: 3
Ingestion completed.
```

Re-running ingest is safe — it uses content-based IDs (`SHA-256` of source + chunk index + text), so existing chunks are upserted, not duplicated.

### 3. Query the system

```bash
uv run python -m src.ask "What is RAG?"
```

Output:

```
Answer:

RAG stands for Retrieval-Augmented Generation. It retrieves relevant context from a
knowledge base and sends that context to a language model to answer questions with
supporting information.
Source: docs/rag.txt
```

The system only answers from the ingested documents. If the answer is not in the context it will say so explicitly.

## Project structure

```
chromadb-rag/
├── docs/               # .txt documents to ingest
├── src/
│   ├── config.py       # Centralized settings
│   ├── ingest.py       # Chunking, embedding, and upserting into ChromaDB
│   └── ask.py          # Query embedding, retrieval, and answer generation
├── chroma_data/        # Persisted ChromaDB vector store (auto-created)
├── .env                # API key (not committed)
└── pyproject.toml
```

## Roadmap

Planned improvements roughly ordered by value/complexity:

| # | Improvement | Why |
|---|---|---|
| 1 | **PDF support** | Most real-world documents are PDFs, not plain text |
| 2 | **Batch processing** | Embed multiple chunks in a single API call to reduce latency and cost |
| 3 | **Skip already-ingested docs** | Avoid re-embedding unchanged files on every ingest run |
| 4 | **Store document hash** | Detect content changes and re-ingest only modified files |
| 5 | **Token-based chunking** | Character splits break mid-word; token-aware splits respect model context boundaries |
| 6 | **FastAPI layer** | Expose ingest and query as HTTP endpoints for integration with other services |
| 7 | **Streaming responses** | Stream Gemini output token-by-token for a more responsive UX |
| 8 | **Clearer citations** | Return page number, line range, and exact snippet alongside the source file |
| 9 | **Retrieval quality metrics** | Measure precision/recall against a labeled eval set to catch regressions |
| 10 | **Reranking** | Add a cross-encoder reranker to reorder top-K results before sending context to the LLM |
