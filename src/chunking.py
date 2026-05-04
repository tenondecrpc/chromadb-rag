"""Token-aware chunking using tiktoken for semantic text splitting."""

import re
import tiktoken

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, ENCODING_NAME


def get_encoder(encoding_name: str = ENCODING_NAME):
    """Get a tiktoken encoder by name."""
    return tiktoken.get_encoding(encoding_name)


def split_into_chunks(
    text: str,
    max_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
    encoding_name: str = ENCODING_NAME,
) -> list[str]:
    """
    Split text into chunks that don't exceed max_tokens.

    Strategy:
      1. Split by paragraphs (\n\n+)
      2. If a paragraph exceeds max_tokens, split by sentences
      3. Add overlap between chunks for context preservation
    """
    encoding = get_encoder(encoding_name)
    paragraphs = re.split(r"\n\n+", text.strip())

    chunks = []
    current_chunk = []
    current_token_count = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_tokens = len(encoding.encode(paragraph))

        # If a single paragraph exceeds the limit, split by sentences
        if paragraph_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_tokens = len(encoding.encode(sentence))

                if current_token_count + sentence_tokens <= max_tokens:
                    current_chunk.append(sentence)
                    current_token_count += sentence_tokens
                else:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sentence]
                    current_token_count = sentence_tokens
        else:
            # Check if adding this paragraph would exceed the limit
            if current_token_count + paragraph_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_token_count = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_token_count += paragraph_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Add overlap between chunks for context preservation
    if overlap_tokens > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap_tokens, encoding)

    return chunks


def _add_overlap(
    chunks: list[str], overlap_tokens: int, encoding
) -> list[str]:
    """Add overlapping text between consecutive chunks."""
    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        prev_tokens = encoding.encode(prev_chunk)

        # Take the last N tokens from the previous chunk
        overlap_start = max(0, len(prev_tokens) - overlap_tokens)
        overlap_token_ids = prev_tokens[overlap_start:]
        overlap_text = encoding.decode(overlap_token_ids)

        # Prepend overlap to current chunk
        new_chunk = overlap_text + "\n\n" + chunks[i]
        overlapped.append(new_chunk)

    return overlapped


def count_tokens(text: str, encoding_name: str = ENCODING_NAME) -> int:
    """Count tokens in a text string."""
    encoding = get_encoder(encoding_name)
    return len(encoding.encode(text))
