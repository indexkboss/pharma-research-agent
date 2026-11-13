
from pharma_research_agent.agent.workflow import create_workflow
from pharma_research_agent.retrieval.pipeline import build_index


def main():
    question = (
        "A patient taking warfarin is prescribed sertraline. "
        "What is the bleeding risk and what monitoring is recommended?"
    )

    print("=" * 70)
    print("COMPLETE AGENT WORKFLOW TEST")
    print("=" * 70)

    # Build the index once.
    print("\nBuilding the corpus index...")

    vector_store, document_count, chunk_count = build_index()

    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")

    # Create the LangGraph workflow.
    print("\nCreating the LangGraph workflow...")

    agent = create_workflow(vector_store)

    # Execute the complete workflow.
    print("\nInvoking the agent...")

    result = agent.invoke(
        {
            "question": question,
            "revision_count": 0,
        }
    )

    # Display the final result.
    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(result["final_answer"])

    # Display workflow diagnostics.
    print("\n" + "=" * 70)
    print("WORKFLOW DIAGNOSTICS")
    print("=" * 70)

    print(f"Subquestions: {len(result.get('subquestions', []))}")
    print(f"Tool calls: {len(result.get('tool_calls', []))}")
    print(f"Evidence items: {len(result.get('evidence', []))}")
    print(f"Revision attempts: {result.get('revision_count', 0)}")
    print(f"Grounded: {result.get('is_grounded', False)}")

    print("\nAgent workflow test completed successfully.")


if __name__ == "__main__":
    main()
