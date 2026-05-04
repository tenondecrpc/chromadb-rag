"""Ingest documents into ChromaDB with embeddings and rich metadata."""

import hashlib
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import track

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    CONVERTED_DIR,
    DOCS_PATH,
    EMBEDDING_MODEL,
)
from src.chunking import split_into_chunks
from src.markdown_converter import (
    convert_documents,
    get_markdown_files,
)

load_dotenv()
console = Console()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Gemini."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [list(e.values) for e in result.embeddings]


def stable_id(source: str, chunk_index: int, text: str) -> str:
    """Generate a stable hash ID for a chunk."""
    raw = f"{source}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_source_url(text: str) -> str | None:
    """Extract source URL from MarkItDown comment if present."""
    match = re.search(r"<!--\s*source:\s*(.+?)\s*-->", text)
    return match.group(1) if match else None


def ingest_documents(convert_first: bool = True):
    """
    Ingest documents into ChromaDB.

    Args:
        convert_first: Whether to run MarkItDown conversion before ingesting
    """
    # Step 1: Convert documents if needed
    if convert_first:
        console.print("[bold cyan]Step 1: Converting documents to Markdown...[/bold cyan]")
        convert_documents(source_dir=DOCS_PATH)
        console.print()

    # Step 2: Get markdown files
    files = get_markdown_files(CONVERTED_DIR)
    if not files:
        console.print("[red]No Markdown files found. Run conversion first or add .md files.[/red]")
        return

    console.print(f"[cyan]Found {len(files)} Markdown files to ingest[/cyan]")

    # Step 3: Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "RAG with Gemini embeddings and tiktoken chunking",
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    # Step 4: Process files
    all_ids = []
    all_documents = []
    all_metadatas = []
    total_chunks = 0

    for file_path in track(files, description="[cyan]Processing files...", console=console):
        text = file_path.read_text(encoding="utf-8")
        source_url = extract_source_url(text)

        # Remove the source URL comment from the text
        if source_url:
            text = re.sub(r"<!--\s*source:\s*.+?\s*-->\n\n", "", text, count=1)

        chunks = split_into_chunks(text)
        file_total = len(chunks)
        total_chunks += file_total

        for index, chunk in enumerate(chunks):
            chunk_id = stable_id(str(file_path), index, chunk)
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_metadatas.append(
                {
                    "source": str(file_path),
                    "chunk_index": index,
                    "total_chunks": file_total,
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "converted_from_url": source_url or "local_file",
                    "chunk_size_tokens": len(chunk.split()),  # Approximate
                }
            )

    console.print(f"[green]Total chunks generated: {total_chunks}[/green]")

    # Step 5: Generate embeddings in batches
    console.print("[cyan]Generating embeddings...[/cyan]")
    embeddings = embed_texts(all_documents)

    # Step 6: Store in ChromaDB
    collection.upsert(
        ids=all_ids,
        documents=all_documents,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    console.print(f"[bold green]Ingested {len(all_documents)} chunks into '{COLLECTION_NAME}'[/bold green]")


if __name__ == "__main__":
    ingest_documents()
