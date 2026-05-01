import hashlib
import os

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DOCS_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [list(e.values) for e in result.embeddings]


def stable_id(source: str, chunk_index: int, text: str) -> str:
    raw = f"{source}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ingest():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Minimal RAG with Gemini embeddings"},
    )

    ids = []
    documents = []
    metadatas = []

    files = list(DOCS_PATH.glob("*.txt"))

    if not files:
        print("No .txt files found in docs/")
        return

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            ids.append(stable_id(str(file_path), index, chunk))
            documents.append(chunk)
            metadatas.append(
                {
                    "source": str(file_path),
                    "chunk_index": index,
                }
            )

    print(f"Files found: {len(files)}")
    print(f"Chunks generated: {len(documents)}")

    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Ingestion completed.")


if __name__ == "__main__":
    ingest()
