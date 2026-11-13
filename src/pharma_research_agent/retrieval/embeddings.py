
import numpy as np

from pharma_research_agent.config import (
    OPENAI_EMBEDDING_MODEL,
    create_client,
)


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Convert a list of texts into embedding vectors.

    Texts are processed in batches to avoid sending
    the entire corpus in one API request.
    """

    if not texts:
        raise ValueError("Cannot generate embeddings for empty input.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    client = create_client()
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        batch_embeddings = [
            item.embedding for item in ordered_data
        ]

        all_embeddings.extend(batch_embeddings)

    return np.asarray(all_embeddings, dtype=np.float32)
