
from pharma_research_agent.agent.researcher import research_subquestions
from pharma_research_agent.agent.answer import generate_draft_answer
from pharma_research_agent.agent.critic import review_answer
from pharma_research_agent.retrieval.pipeline import build_index


def main():
    question = (
        "A patient taking warfarin is prescribed sertraline. "
        "What is the bleeding risk and what monitoring is recommended?"
    )

    subquestions = [
        (
            "What evidence describes the bleeding risk when "
            "sertraline is used concomitantly with warfarin?"
        ),
        (
            "What monitoring and management recommendations are "
            "supported for patients taking warfarin who start sertraline?"
        ),
    ]

    print("=" * 60)
    print("ANSWER GENERATION AND CRITIC TEST")
    print("=" * 60)

    # 1. Build the index.
    print("\nBuilding the corpus index...")
    vector_store, document_count, chunk_count = build_index()

    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")

    # 2. Research the question.
    print("\n[1/3] Researching the question...")

    research_result = research_subquestions(
        question=question,
        subquestions=subquestions,
        vector_store=vector_store,
    )

    state = {
        "question": question,
        "subquestions": subquestions,
        "evidence": research_result["evidence"],
        "tool_calls": research_result["tool_calls"],
    }

    print(f"Evidence items collected: {len(state['evidence'])}")

    # 3. Generate the preliminary answer.
    print("\n[2/3] Generating the draft answer...")

    answer_result = generate_draft_answer(state)
    state.update(answer_result)

    print("\nDRAFT ANSWER")
    print("-" * 60)
    print(state["draft_answer"])

    # 4. Review the draft.
    print("\n[3/3] Reviewing the answer...")

    review_result = review_answer(state)
    state.update(review_result)

    print("\nCRITIC REVIEW")
    print("-" * 60)
    print(f"Grounded: {state['is_grounded']}")
    print(f"Critique: {state['critique']}")

    print("\nIssues identified:")
    if state["review_issues"]:
        for issue in state["review_issues"]:
            print(f"- {issue}")
    else:
        print("- No issues reported by the critic.")

    print("\nAnswer and critic test completed successfully.")


if __name__ == "__main__":
    main()
