
from pharma_research_agent.agent.planner import plan_question


def main():
    question = (
        "A patient taking warfarin is prescribed sertraline. "
        "What is the bleeding risk and what monitoring is recommended?"
    )

    initial_state = {
        "question": question,
    }

    print("=" * 60)
    print("PLANNER TEST")
    print("=" * 60)

    print("\nOriginal question:")
    print(question)

    print("\nGenerating research plan...")

    result = plan_question(initial_state)

    print("\nGenerated subquestions:")

    for index, subquestion in enumerate(
        result["subquestions"],
        start=1,
    ):
        print(f"{index}. {subquestion}")

    print("\nPlanner test completed successfully.")


if __name__ == "__main__":
    main()
