
from pharma_research_agent.agent.tools import create_tool_registry
from pharma_research_agent.retrieval.pipeline import build_index


def main():
    print("=" * 60)
    print("LOCAL TOOLS TEST")
    print("=" * 60)

    # Build the semantic index once.
    print("\nBuilding the corpus index...")
    vector_store, document_count, chunk_count = build_index()

    print(f"Documents indexed: {document_count}")
    print(f"Chunks indexed: {chunk_count}")

    # Create the two local tools.
    tools = create_tool_registry(vector_store)

    # Test 1: semantic search.
    print("\n[1/2] Testing search_corpus...")

    search_result = tools["search_corpus"](
        query="What are the contraindications of warfarin?",
        top_k=3,
    )

    print(f"Number of results: {len(search_result['results'])}")

    for index, result in enumerate(search_result["results"], start=1):
        print(f"\nResult {index}")
        print(f"Source: {result['source']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Passage: {result['text'][:250]}...")

    # Test 2: reading an original source document.
    print("\n[2/2] Testing read_source_document...")

    document_result = tools["read_source_document"](
        source="dailymed-warfarin.md"
    )

    if "error" in document_result:
        print(f"Document reading failed: {document_result['error']}")
    else:
        print(f"Source: {document_result['source']}")
        print(f"Content length: {len(document_result['content'])} characters")
        print("\nFirst 300 characters:")
        print(document_result["content"][:300])

    print("\nLocal tools test completed.")


if __name__ == "__main__":
    main()
