from pharma_research_agent.agent.critic import review_answer
from pharma_research_agent.retrieval.pipeline import build_index


def main():

    print("=" * 70)
    print("ADVERSARIAL CRITIC TEST")
    print("=" * 70)

    # 1. Build the real corpus index
    print("\nBuilding document index...")

    vector_store, document_count, chunk_count = build_index()

    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")

    # 2. Retrieve actual evidence
    question = (
        "At what eGFR level is metformin contraindicated?"
    )

    evidence = vector_store.search(
        "Metformin contraindication eGFR below 30 mL/min/1.73 m2",
        top_k=5,
    )

    if not evidence:
        print("No evidence retrieved. Test cannot continue.")
        return

    print("\nRetrieved sources:")

    for item in evidence:
        print(f"- {item['source']}")

    # 3. Create a deliberately incorrect answer
    draft_answer = """
    According to the supplied evidence, metformin is contraindicated
    only when eGFR falls below 20 mL/min/1.73 m2.

    Therefore, a patient with an eGFR of 25 mL/min/1.73 m2
    can continue metformin without any renal-related concern.

    [dailymed-metformin.md]
    """

    # 4. Ask the critic to review the answer
    state = {
        "question": question,
        "subquestions": [
            "Identify the renal threshold for metformin contraindication."
        ],
        "evidence": evidence,
        "draft_answer": draft_answer,
    }

    print("\nReviewing deliberately incorrect answer...")

    result = review_answer(state)

    # 5. Display the critic's assessment
    print("\n" + "=" * 70)
    print("CRITIC RESULTS")
    print("=" * 70)

    print(f"\nGrounded: {result['is_grounded']}")

    print("\nCritique:")
    print(result["critique"])

    print("\nReview issues:")

    for issue in result.get("review_issues", []):
        print(f"- {issue}")

    # 6. Interpret the result
    print("\n" + "=" * 70)

    if result["is_grounded"] is False:
        print("PASS: The critic rejected the unsupported answer.")

    else:
        print(
            "WARNING: The critic accepted the deliberately "
            "incorrect answer."
        )
        print(
            "The critic prompt or evidence-review process "
            "needs improvement."
        )


if __name__ == "__main__":
    main()
