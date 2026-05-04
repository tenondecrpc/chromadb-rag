from pathlib import Path

# Paths
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "chromadb_rag_collection"
DOCS_PATH = Path("docs")

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
ENCODING_NAME = "cl100k_base"

# Retrieval
TOP_K = 3

# Pipeline settings
CONVERTED_DIR = Path("converted")
SUPPORTED_DOC_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
