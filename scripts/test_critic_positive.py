
from pharma_research_agent.agent.critic import review_answer
from pharma_research_agent.retrieval.pipeline import build_index


def main():
    print("=" * 70)
    print("POSITIVE CRITIC TEST")
    print("=" * 70)

    # 1. Build the real corpus index
    print("\nBuilding document index...")

    vector_store, document_count, chunk_count = build_index()

    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")

    # 2. Retrieve evidence from the metformin document
    question = "At what eGFR level is metformin contraindicated?"

    evidence = vector_store.search(
        "Metformin renal impairment eGFR contraindication "
        "below 30 initiation 30 to 45",
        top_k=8,
    )

    if not evidence:
        print("No evidence retrieved. Test cannot continue.")
        return

    print("\nRetrieved sources:")
    for item in evidence:
        print(f"- {item['source']}")

    # 3. Provide an answer that accurately reflects the evidence
    draft_answer = """
    Metformin is contraindicated in patients with severe renal
    impairment, defined as an eGFR below 30 mL/min/1.73 m².

    Initiating metformin is not recommended when eGFR is between
    30 and 45 mL/min/1.73 m². For patients already taking metformin
    whose eGFR falls below 45 mL/min/1.73 m², the benefits and risks
    of continuing treatment should be reassessed.

    Therefore, an eGFR of 25 mL/min/1.73 m² is below the
    contraindication threshold.

    [dailymed-metformin.md]
    """

    # 4. Ask the critic to review the answer
    state = {
        "question": question,
        "subquestions": [
            "Identify the renal threshold for metformin contraindication.",
            "Identify the additional eGFR precautions."
        ],
        "evidence": evidence,
        "draft_answer": draft_answer,
    }

    print("\nReviewing the evidence-supported answer...")

    result = review_answer(state)

    # 5. Display the assessment
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

    if result["is_grounded"] is True:
        print("PASS: The critic accepted the evidence-supported answer.")
    else:
        print("WARNING: The critic rejected the correct answer.")
        print("Inspect its review issues to understand why.")


if __name__ == "__main__":
    main()
