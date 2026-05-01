from pathlib import Path

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "minimal_rag"

DOCS_PATH = Path("docs")

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 3