import os
import sys

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K,
)


load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def embed_query(query: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(result.embeddings[0].values)


def retrieve_context(question: str, top_k: int = TOP_K):
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
    context_blocks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, document in enumerate(documents):
        metadata = metadatas[i]
        distance = distances[i]

        context_blocks.append(
            f"""
Source: {metadata["source"]}
Chunk index: {metadata["chunk_index"]}
Distance: {distance}

Content:
{document}
""".strip()
        )

    return "\n\n---\n\n".join(context_blocks)


def answer_question(question: str) -> str:
    results = retrieve_context(question)
    context = build_context(results)

    prompt = f"""
You are a RAG assistant.

Rules:
1. Answer only using the provided context.
2. If the answer is not in the context, say: "I do not have enough information in the provided documents."
3. Cite the source file names used in your answer.
4. Be concise.

Context:
{context}

User question:
{question}
""".strip()

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python -m src.ask "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer = answer_question(question)

    print("\nAnswer:\n")
    print(answer)
