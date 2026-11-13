
from pharma_research_agent.ingestion.loader import (
    load_markdown_documents,
)

from pharma_research_agent.ingestion.chunker import (
    chunk_documents,
)

from pharma_research_agent.retrieval.embeddings import (
    embed_texts,
)

from pharma_research_agent.retrieval.vector_store import (
    InMemoryVectorStore,
)


def build_index() -> tuple[InMemoryVectorStore, int, int]:
    """
    Build the complete in-memory retrieval index.

    Returns:
        vector_store: The searchable vector store.
        document_count: Number of loaded documents.
        chunk_count: Number of generated chunks.
    """

    print("Step 1/4: Loading documents...")

    documents = load_markdown_documents()

    print(f"Loaded {len(documents)} documents.")

    print("\nStep 2/4: Splitting documents into chunks...")

    chunks = chunk_documents(documents)

    print(f"Generated {len(chunks)} chunks.")

    print("\nStep 3/4: Generating embeddings...")

    texts = [chunk.text for chunk in chunks]

    embeddings = embed_texts(texts)

    print(f"Generated {len(embeddings)} embeddings.")

    print("\nStep 4/4: Building the vector store...")

    vector_store = InMemoryVectorStore(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Vector store successfully created.")

    return vector_store, len(documents), len(chunks)
