import json
from pathlib import Path

from pharma_research_agent.agent.workflow import create_workflow
from pharma_research_agent.retrieval.pipeline import build_index


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "evaluation_results.json"


# The four official pharmaceutical evaluation questions
QUESTIONS = [
    {
        "id": 1,
        "question": (
            "What are the main contraindications of warfarin?"
        ),
    },
    {
        "id": 2,
        "question": (
            "Compare the contraindications of warfarin and apixaban."
        ),
    },
    {
        "id": 3,
        "question": (
            "A patient taking warfarin is prescribed sertraline. "
            "What are the potential bleeding risks and what "
            "monitoring should be considered?"
        ),
    },
    {
        "id": 4,
        "question": (
            "A patient has an eGFR of 25 and is taking metformin. "
            "Is metformin contraindicated in this situation, "
            "and what renal threshold is relevant?"
        ),
    },
]


def main():

    print("=" * 70)
    print("PHARMACEUTICAL RESEARCH AGENT - EVALUATION")
    print("=" * 70)

    # Build the document index once for all questions
    print("\nBuilding the corpus index...")

    vector_store, document_count, chunk_count = build_index()

    print(f"Documents indexed: {document_count}")
    print(f"Chunks indexed: {chunk_count}")

    # Create one agent and reuse it
    agent = create_workflow(vector_store)

    results = []

    for item in QUESTIONS:

        question_id = item["id"]
        question = item["question"]

        print("\n" + "=" * 70)
        print(f"QUESTION {question_id}/4")
        print("=" * 70)
        print(question)

        # Execute the complete LangGraph workflow
        result = agent.invoke(
            {
                "question": question,
                "revision_count": 0,
            }
        )

        # Extract source filenames from the evidence
        sources = sorted({
            evidence["source"]
            for evidence in result.get("evidence", [])
            if evidence.get("source")
        })

        evaluation = {
            "id": question_id,
            "question": question,
            "answer": result.get("final_answer", ""),
            "grounded": result.get("is_grounded", False),
            "sources": sources,
            "subquestions": result.get("subquestions", []),
            "review_issues": result.get("review_issues", []),
            "revision_count": result.get("revision_count", 0),
        }

        results.append(evaluation)

        # Display the evaluation
        print("\n--- FINAL ANSWER ---")
        print(evaluation["answer"])

        print("\n--- GROUNDING ASSESSMENT ---")
        print(evaluation["grounded"])

        print("\n--- SOURCES ---")

        for source in sources:
            print(f"- {source}")

        print("\n--- REVIEW ISSUES ---")

        if evaluation["review_issues"]:
            for issue in evaluation["review_issues"]:
                print(f"- {issue}")
        else:
            print("No issues reported by the critic.")

    # Save all results for later analysis
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Questions evaluated: {len(results)}")
    print(f"Results saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
