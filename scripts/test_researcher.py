from pharma_research_agent.agent.researcher import research_subquestions
from pharma_research_agent.retrieval.pipeline import build_index


def main():
    question = (
        "A patient taking warfarin is prescribed sertraline. "
        "What evidence describes the bleeding risk?"
    )

    print("=" * 60)
    print("DYNAMIC TOOL SELECTION TEST")
    print("=" * 60)

    print("\nBuilding the corpus index...")
    vector_store, document_count, chunk_count = build_index()

    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")

    # We use one research question to test the tool-selection mechanism.
    subquestions = [
        "What evidence describes the bleeding risk when sertraline "
        "is used concomitantly with warfarin?"
    ]

    print("\nStarting agent research...")

    result = research_subquestions(
        question=question,
        subquestions=subquestions,
        vector_store=vector_store,
    )

    print("\n" + "=" * 60)
    print("RESEARCH RESULTS")
    print("=" * 60)

    print(f"\nNumber of tool calls: {len(result['tool_calls'])}")

    print("\nTools selected by the model:")
    for call in result["tool_calls"]:
        print(f"- Tool: {call['tool']}")
        print(f"  Arguments: {call['arguments']}")

    print(f"\nEvidence passages/documents collected: {len(result['evidence'])}")

    sources = sorted({
        item["source"]
        for item in result["evidence"]
        if item.get("source")
    })

    print("\nSources used:")
    for source in sources:
        print(f"- {source}")

    if not result["tool_calls"]:
        raise RuntimeError(
            "The model did not request any tool calls. "
            "Check whether the configured model supports tool calling."
        )

    if not result["evidence"]:
        raise RuntimeError(
            "The tools ran, but no evidence was collected."
        )

    print("\nDynamic tool selection test completed successfully.")


if __name__ == "__main__":
    main()
