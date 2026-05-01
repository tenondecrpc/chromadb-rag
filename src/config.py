from pathlib import Path

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "chromadb_rag_collection"

DOCS_PATH = Path("docs")

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 3

EMBED_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 500
MAX_WORKERS = 5
MAX_RETRIES = 4