
import json

from pharma_research_agent.config import (
    OPENAI_MODEL,
    create_client,
)

from pharma_research_agent.agent.state import AgentState


PLANNER_SYSTEM_PROMPT = """
You are the planning component of a pharmaceutical research assistant.

Your task is to decompose the user's question into smaller research
questions that can be investigated using a local pharmaceutical corpus.

Rules:
1. Do not answer the medical question.
2. Identify the distinct information needs in the question.
3. Create focused, non-redundant research questions.
4. Preserve important drug names, conditions, and comparisons.
5. For compound questions, separate the different aspects that
   require evidence.
6. Do not invent medical facts.
7. Produce between 1 and 5 subquestions.

Return a valid JSON object with exactly this structure:

{
    "subquestions": [
        "First research question",
        "Second research question"
    ]
}
"""


def plan_question(state: AgentState) -> dict:
    """
    Decompose the user's question into research subquestions.

    This function is intended to be used as a LangGraph node.
    """

    question = state["question"].strip()

    if not question:
        raise ValueError("The user's question cannot be empty.")

    client = create_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The planner returned an empty response.")

    try:
        parsed_response = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The planner did not return valid JSON."
        ) from exc

    subquestions = parsed_response.get("subquestions")

    if not isinstance(subquestions, list):
        raise ValueError(
            "The planner response must contain a list of subquestions."
        )

    cleaned_subquestions = [
        item.strip()
        for item in subquestions
        if isinstance(item, str) and item.strip()
    ]

    if not cleaned_subquestions:
        raise ValueError(
            "The planner did not generate any valid subquestions."
        )

    if len(cleaned_subquestions) > 5:
        cleaned_subquestions = cleaned_subquestions[:5]

    return {
        "subquestions": cleaned_subquestions,
    }
