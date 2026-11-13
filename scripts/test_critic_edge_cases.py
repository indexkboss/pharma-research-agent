
from pharma_research_agent.agent.critic import review_answer
from pharma_research_agent.retrieval.pipeline import build_index


def run_case(name, question, evidence, draft_answer):
    """Run one critic test and display its assessment."""

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    state = {
        "question": question,
        "subquestions": [
            "Identify the evidence supporting the answer."
        ],
        "evidence": evidence,
        "draft_answer": draft_answer,
    }

    result = review_answer(state)

    print(f"\nGrounded: {result['is_grounded']}")
    print("\nCritique:")
    print(result["critique"])

    print("\nReview issues:")
    for issue in result.get("review_issues", []):
        print(f"- {issue}")

    passed = result["is_grounded"] is False

    print("\nTest result:", "PASS" if passed else "WARNING")

    return passed


def main():
    print("=" * 70)
    print("CRITIC EDGE-CASE TESTS")
    print("=" * 70)

    # Build the index once for the evidence-based scenarios.
    print("\nBuilding document index...")

    vector_store, document_count, chunk_count = build_index()

    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")

    question = "At what eGFR level is metformin contraindicated?"

    # CASE 1: No evidence at all.
    no_evidence = []

    unsupported_answer = """
    Metformin is contraindicated only below an eGFR of 20.
    A patient with an eGFR of 25 can safely continue treatment.
    [dailymed-metformin.md]
    """

    result_1 = run_case(
        "CASE 1 — NO EVIDENCE",
        question,
        no_evidence,
        unsupported_answer,
    )

    # CASE 2: Evidence is available, but it concerns aspirin.
    irrelevant_evidence = vector_store.search(
        "Aspirin contraindications, bleeding risks and precautions",
        top_k=5,
    )

    irrelevant_answer = """
    Metformin is contraindicated only below an eGFR of 20.
    [dailymed-metformin.md]
    """

    result_2 = run_case(
        "CASE 2 — IRRELEVANT EVIDENCE",
        question,
        irrelevant_evidence,
        irrelevant_answer,
    )

    # CASE 3: Relevant evidence, but the answer cites the wrong document.
    relevant_evidence = vector_store.search(
        "Metformin renal impairment contraindication eGFR below 30",
        top_k=5,
    )

    wrong_citation_answer = """
    Metformin is contraindicated when eGFR is below
    30 mL/min/1.73 m².
    [dailymed-warfarin.md]
    """

    result_3 = run_case(
        "CASE 3 — INCORRECT CITATION",
        question,
        relevant_evidence,
        wrong_citation_answer,
    )

    # Final summary.
    results = [result_1, result_2, result_3]

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("PASS: All critic edge-case tests passed.")
    else:
        print(
            "WARNING: At least one scenario was accepted "
            "when it should have been rejected."
        )


if __name__ == "__main__":
    main()
