
from pharma_research_agent.retrieval.pipeline import build_index


def main():
    print("=" * 60)
    print("PHARMACEUTICAL RAG RETRIEVAL TEST")
    print("=" * 60)

    # Build the retrieval index.
    vector_store, document_count, chunk_count = build_index()

    print("\n" + "=" * 60)
    print("INDEX SUMMARY")
    print("=" * 60)

    print(f"Documents loaded: {document_count}")
    print(f"Chunks indexed: {chunk_count}")
    print(f"Vectors stored: {vector_store.chunk_count}")

    # Test a question from the assignment.
    question = "What are the main contraindications for warfarin?"

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    print(f"Question: {question}")

    results = vector_store.search(
        query=question,
        top_k=3,
    )

    print("\nTop 3 retrieved passages:")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}")

    print("\nRAG retrieval test completed successfully.")


if __name__ == "__main__":
    main()
