import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from src.config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCS_PATH,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    MAX_RETRIES,
    MAX_WORKERS,
    UPSERT_BATCH_SIZE,
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


def stable_id(source: str, chunk_index: int, text: str) -> str:
    raw = f"{source}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _embed_batch(texts: list[str], batch_index: int) -> tuple[int, list[list[float]]]:
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return batch_index, [list(e.values) for e in result.embeddings]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Batch {batch_index} failed after {MAX_RETRIES} attempts: {e}") from e
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def ingest():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Minimal RAG with Gemini embeddings"},
    )

    files = list(DOCS_PATH.glob("*.txt"))
    if not files:
        print("No .txt files found in docs/")
        return

    # Phase 1: chunk all files
    all_ids = []
    all_documents = []
    all_metadatas = []

    t_chunk_start = time.perf_counter()
    for file_path in tqdm(files, desc="Chunking"):
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for index, chunk in enumerate(chunks):
            all_ids.append(stable_id(str(file_path), index, chunk))
            all_documents.append(chunk)
            all_metadatas.append({"source": str(file_path), "chunk_index": index})

    t_chunk_end = time.perf_counter()
    total_chunks = len(all_documents)
    print(f"Files: {len(files)} | Chunks: {total_chunks}")

    # Phase 2: embed in parallel batches
    t_embed_start = time.perf_counter()
    batches = [
        all_documents[i : i + EMBED_BATCH_SIZE]
        for i in range(0, total_chunks, EMBED_BATCH_SIZE)
    ]
    ordered_embeddings: list[list[list[float]] | None] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_embed_batch, batch, i): i for i, batch in enumerate(batches)}
        with tqdm(total=total_chunks, desc="Embedding") as pbar:
            for future in as_completed(futures):
                batch_index, embeddings = future.result()
                ordered_embeddings[batch_index] = embeddings
                pbar.update(len(batches[batch_index]))

    t_embed_end = time.perf_counter()
    flat_embeddings = [emb for batch in ordered_embeddings for emb in batch]  # type: ignore[union-attr]

    # Phase 3: upsert to ChromaDB in batches
    t_upsert_start = time.perf_counter()
    with tqdm(total=total_chunks, desc="Upserting") as pbar:
        for i in range(0, total_chunks, UPSERT_BATCH_SIZE):
            end = i + UPSERT_BATCH_SIZE
            collection.upsert(
                ids=all_ids[i:end],
                documents=all_documents[i:end],
                embeddings=flat_embeddings[i:end],
                metadatas=all_metadatas[i:end],
            )
            pbar.update(min(UPSERT_BATCH_SIZE, total_chunks - i))

    t_upsert_end = time.perf_counter()
    t_total = t_upsert_end - t_chunk_start

    print(
        f"Ingestion completed. "
        f"Chunk: {t_chunk_end - t_chunk_start:.2f}s | "
        f"Embed: {t_embed_end - t_embed_start:.2f}s | "
        f"Upsert: {t_upsert_end - t_upsert_start:.2f}s | "
        f"Total: {t_total:.2f}s"
    )


if __name__ == "__main__":
    ingest()
