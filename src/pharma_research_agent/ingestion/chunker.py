
from dataclasses import dataclass

from pharma_research_agent.ingestion.loader import SourceDocument


@dataclass(frozen=True)
class TextChunk:
    """Represents one text passage and its source."""

    chunk_id: str
    source: str
    chunk_index: int
    text: str


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """
    Split a document into overlapping text chunks.

    The splitter prefers paragraph, line, and sentence boundaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."
        )

    text = document.content.strip()

    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            minimum_boundary = start + int(
                (end - start) * 0.55
            )

            boundaries = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]

            valid_boundaries = [
                boundary
                for boundary in boundaries
                if boundary >= minimum_boundary
            ]

            if valid_boundaries:
                end = max(valid_boundaries)

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{document.source}#chunk-{chunk_index}",
                    source=document.source,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

            chunk_index += 1

        if end >= len(text):
            break

        next_start = end - chunk_overlap

        start = max(next_start, start + 1)

    return chunks


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Split multiple documents into chunks."""

    all_chunks = []

    for document in documents:
        document_chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(document_chunks)

    if not all_chunks:
        raise ValueError("No text chunks were generated.")

    return all_chunks
