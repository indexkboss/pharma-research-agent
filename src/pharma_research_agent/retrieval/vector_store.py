
import numpy as np

from pharma_research_agent.ingestion.chunker import TextChunk
from pharma_research_agent.retrieval.embeddings import embed_texts


class InMemoryVectorStore:
    """A simple vector store using NumPy and cosine similarity."""

    def __init__(
        self,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
    ):
        if not chunks:
            raise ValueError("The vector store cannot be empty.")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "The number of chunks must match the number of vectors."
            )

        self.chunks = chunks
        self.embeddings = embeddings

        # Precompute the norm of each document vector.
        self.embedding_norms = np.linalg.norm(
            self.embeddings,
            axis=1,
        )

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top_k most similar chunks for a question.
        """

        if not query.strip():
            raise ValueError("The search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        # Convert the user's question into an embedding.
        query_embedding = embed_texts([query])[0]

        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            raise ValueError("The query embedding has zero magnitude.")

        # Calculate cosine similarity against every document vector.
        denominators = self.embedding_norms * query_norm

        scores = np.divide(
            self.embeddings @ query_embedding,
            denominators,
            out=np.zeros_like(denominators),
            where=denominators != 0,
        )

        # Retrieve the indices of the highest similarity scores.
        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in best_indices:
            chunk = self.chunks[int(index)]

            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "score": float(scores[index]),
                }
            )

        return results
