"""Query ChromaDB and generate answers with Gemini, with interactive Rich UI."""

import os
import sys

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K,
)

load_dotenv()
console = Console()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def embed_query(query: str) -> list[float]:
    """Generate embedding for a query string."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(result.embeddings[0].values)


def retrieve_context(question: str, top_k: int = TOP_K):
    """Retrieve relevant documents from ChromaDB."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


def build_context(results) -> str:
    """Build a structured context string from retrieval results."""
    context_blocks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, document in enumerate(documents):
        metadata = metadatas[i]
        distance = distances[i]

        block = f"""--- Relevant Information #{i + 1} ---
Source: {metadata['file_name']}
Chunk: {metadata['chunk_index']} / {metadata['total_chunks']}
Distance: {distance:.4f}
Origin: {metadata['converted_from_url']}

Content:
{document}"""
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def answer_question(question: str) -> str:
    """Answer a question using RAG."""
    results = retrieve_context(question)
    context = build_context(results)

    prompt = f"""You are a RAG assistant. Answer the user's question using only the provided context.

Rules:
1. Use ONLY the information in the provided context.
2. If the answer is not in the context, say: "I do not have enough information in the provided documents."
3. Cite the source file names and chunk numbers used in your answer.
4. Be concise but complete.

Context:
{context}

User question:
{question}

Answer:""".strip()

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    return response.text


def interactive_mode():
    """Run an interactive query loop with Rich UI."""
    console.print(Panel.fit(
        "[bold green]RAG Interactive Assistant[/bold green]\n"
        "[dim]Type your question or 'exit' to quit[/dim]",
        title="ChromaDB RAG",
        border_style="green",
    ))

    while True:
        console.print()
        query = console.input("[bold blue]? [/bold blue]")

        if query.lower().strip() in {"exit", "quit", "salir", "q"}:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if not query.strip():
            console.print("[red]Please enter a valid question.[/red]")
            continue

        console.print(Panel(f"[bold yellow]Question:[/bold yellow] {query}", border_style="yellow"))

        try:
            answer = answer_question(query)
            console.print(Panel(
                Markdown(answer),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            ))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def single_query(question: str):
    """Answer a single question and print the result."""
    console.print(Panel(f"[bold yellow]Question:[/bold yellow] {question}", border_style="yellow"))

    try:
        answer = answer_question(question)
        console.print(Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        interactive_mode()
    else:
        question = " ".join(sys.argv[1:])
        single_query(question)
