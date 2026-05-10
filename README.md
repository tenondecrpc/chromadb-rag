# chromadb-rag

A Retrieval-Augmented Generation (RAG) system built with ChromaDB and Google Gemini. It supports multiple document formats, uses token-aware chunking, and provides an interactive CLI with Rich UI.

## Features

- **Multi-format ingestion**: Converts `.txt`, `.pdf`, `.docx`, `.html`, and URLs to Markdown via [MarkItDown](https://github.com/microsoft/markitdown)
- **Token-aware chunking**: Uses `tiktoken` to split text at semantic boundaries (paragraphs → sentences) with configurable overlap
- **Rich metadata**: Every chunk stores source file, chunk index, total chunks, and origin
- **Interactive CLI**: Query with a beautiful terminal UI using [Rich](https://github.com/Textualize/rich)
- **Modular pipeline**: Run conversion, ingestion, and query independently or in one go

## How it works

```
docs/*.{txt,pdf,docx,html}  →  MarkItDown  →  converted/*.md
                                    ↓
                         chunk (tiktoken)  →  embed (Gemini)  →  ChromaDB (local)
                                                                              ↓
question  →  embed  →  similarity search  →  context + metadata  →  Gemini LLM  →  answer
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
| `CHUNK_SIZE` | `800` | Maximum tokens per chunk |
| `CHUNK_OVERLAP` | `120` | Token overlap between consecutive chunks |
| `ENCODING_NAME` | `cl100k_base` | tiktoken encoding for token counting |
| `TOP_K` | `3` | Number of chunks retrieved per query |
| `DOCS_PATH` | `docs/` | Directory with source documents |
| `CONVERTED_DIR` | `converted/` | Directory for MarkItDown output |

## Usage

### CLI Commands

```bash
# Run the full pipeline: convert → ingest → interactive query
uv run python -m main run

# Or run each step independently:
uv run python -m main convert    # Convert docs to Markdown
uv run python -m main ingest     # Chunk and ingest into ChromaDB
uv run python -m main query      # Start interactive query mode
uv run python -m main help       # Show help
```

### 1. Add your documents

Place files in the `docs/` folder. Supported formats:
- `.txt` — Plain text
- `.pdf` — PDF documents
- `.docx` — Word documents
- `.html`, `.htm` — Web pages
- `.md` — Markdown (passthrough)

The system ships with three example files:

```
docs/
  rag.txt
  chromadb.txt
  aws.txt
```

### 2. Convert documents

Converts all supported files in `docs/` to Markdown using MarkItDown:

```bash
uv run python -m main convert
```

Output:

```
=== Document Conversion ===
Found 3 files to convert in docs
Converting aws.txt...
Saved: converted/aws.md
Converting chromadb.txt...
Saved: converted/chromadb.md
Converting rag.txt...
Saved: converted/rag.md
Total converted: 3 files
```

### 3. Ingest documents

Chunks the Markdown files using tiktoken, generates embeddings via Gemini, and upserts them into ChromaDB:

```bash
uv run python -m main ingest
```

Output:

```
=== Ingestion ===
Found 3 Markdown files to ingest
Processing files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Total chunks generated: 3
Generating embeddings...
Ingested 3 chunks into 'chromadb_rag_collection'
```

Re-running ingest is safe — it uses content-based IDs (`SHA-256` of source + chunk index + text), so existing chunks are upserted, not duplicated.

### 4. Query the system

**Interactive mode** (recommended):

```bash
uv run python -m main query
```

```
╭────────── RAG Interactive Assistant ──────────╮
│ Type your question or 'exit' to quit          │
╰───────────────────────────────────────────────╯

? What is RAG?
╭─ Question ────────────────────────────────────╮
│ What is RAG?                                  │
╰───────────────────────────────────────────────╯
╭─ Answer ──────────────────────────────────────╮
│ RAG stands for Retrieval-Augmented Generation.│
│ It retrieves relevant context from a knowledge│
│ base and sends that context to a language     │
│ model to answer questions.                    │
│                                               │
│ Source: rag.md, Chunk: 0 / 1                  │
╰───────────────────────────────────────────────╯
```

**Single query** (non-interactive):

```bash
uv run python -m src.ask "What is RAG?"
```

The system only answers from the ingested documents. If the answer is not in the context it will say so explicitly.

## Project structure

```
chromadb-rag/
├── docs/                    # Source documents (.txt, .pdf, .docx, .html)
├── converted/               # Markdown files generated by MarkItDown
├── src/
│   ├── config.py            # Centralized configuration
│   ├── chunking.py          # Token-aware text splitting (tiktoken)
│   ├── markdown_converter.py # MarkItDown integration
│   ├── ingest.py            # Chunking, embedding, and upserting
│   ├── ask.py               # Query embedding, retrieval, and generation
│   └── pipeline.py          # Pipeline orchestrator
├── chroma_data/             # Persisted ChromaDB vector store (auto-created)
├── main.py                  # CLI entry point with subcommands
├── .env                     # API key (not committed)
└── pyproject.toml
```

## Roadmap

Planned improvements roughly ordered by value/complexity:

| # | Improvement | Status | Why |
|---|---|---|---|
| 1 | **Multi-format document support** | ✅ Done | Support PDF, DOCX, HTML via MarkItDown |
| 2 | **Token-aware chunking** | ✅ Done | Split at semantic boundaries with tiktoken |
| 3 | **Rich metadata** | ✅ Done | Track source, chunk index, origin per document |
| 4 | **Interactive CLI** | ✅ Done | Beautiful terminal UI with Rich |
| 5 | **Modular pipeline** | ✅ Done | Independent convert/ingest/query steps |
| 6 | **Batch processing** | 🔲 Pending | Embed multiple chunks in a single API call |
| 7 | **Skip already-ingested docs** | 🔲 Pending | Avoid re-embedding unchanged files |
| 8 | **Store document hash** | 🔲 Pending | Detect content changes and re-ingest only modified files |
| 9 | **FastAPI layer** | 🔲 Pending | Expose ingest and query as HTTP endpoints |
| 10 | **Streaming responses** | 🔲 Pending | Stream Gemini output token-by-token |
| 11 | **Metadata filtering** | 🔲 Pending | Filter retrieval by file name, source URL, or other metadata fields |
| 12 | **Retrieval quality metrics** | 🔲 Pending | Measure precision/recall against a labeled eval set |
| 13 | **Reranking** | 🔲 Pending | Add a cross-encoder reranker to reorder top-K results |
| 14 | **Contextual retrieval** | 🔲 Pending | Prepend an LLM-generated context summary to each chunk before embedding to improve retrieval accuracy ([Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)) |
