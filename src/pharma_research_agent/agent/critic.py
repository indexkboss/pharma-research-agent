
import json

from pharma_research_agent.agent.answer import format_evidence
from pharma_research_agent.agent.state import AgentState
from pharma_research_agent.config import OPENAI_MODEL, create_client


CRITIC_SYSTEM_PROMPT = """
You are an independent evidence critic for a pharmaceutical
research assistant.

Your task is to evaluate whether a draft answer is adequately
supported by the supplied local pharmaceutical evidence.

Evaluate these points carefully:

1. Grounding:
   Are the medical and clinical claims supported by the evidence?

2. Citation accuracy:
   Do the cited source filenames exist in the supplied evidence?
   Does the cited evidence actually support the associated claims?

3. Missing information:
   Does the draft answer all important parts of the user's question?
   Does it acknowledge information that the corpus does not provide?

4. Clinical ambiguity:
   Are there vague, misleading, or clinically ambiguous statements?
   Are important qualifications, conditions, or thresholds missing?

5. Unsupported claims:
   Does the draft introduce recommendations, statistics,
   mechanisms, or medical facts that are absent from the evidence?

Be strict. A plausible medical statement is not necessarily
a supported statement.

Do not use external knowledge to justify unsupported claims.

Return a valid JSON object with exactly these fields:

{
    "is_grounded": true,
    "issues": [],
    "unsupported_claims": [],
    "clinical_ambiguities": [],
    "suggested_corrections": [],
    "critique": "A concise explanation of your assessment."
}

Set is_grounded to false if important claims are unsupported,
citations are inaccurate, or clinically significant ambiguity
remains.

Set is_grounded to true only when the answer is adequately
supported and its important limitations are acknowledged.
"""


def review_answer(state: AgentState) -> dict:
    """Review a draft answer against the retrieved evidence."""

    draft = state.get("draft_answer", "")
    evidence = state.get("evidence", [])

    if not draft.strip():
        raise ValueError("There is no draft answer to review.")

    formatted_evidence = format_evidence(evidence)

    messages = [
        {
            "role": "system",
            "content": CRITIC_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{state['question']}\n\n"
                f"Draft answer:\n{draft}\n\n"
                f"Available evidence:\n{formatted_evidence}\n\n"
                "Review the draft and return the required JSON."
            ),
        },
    ]

    client = create_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The critic returned an empty response.")

    try:
        review = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The critic returned invalid JSON.") from exc

    if not isinstance(review.get("is_grounded"), bool):
        raise ValueError("The critic did not return a valid is_grounded value.")

    issues = review.get("issues", [])
    unsupported = review.get("unsupported_claims", [])
    ambiguities = review.get("clinical_ambiguities", [])
    corrections = review.get("suggested_corrections", [])

    # Validate the structure instead of silently accepting malformed output.
    if not all(
        isinstance(items, list)
        for items in [issues, unsupported, ambiguities, corrections]
    ):
        raise ValueError("The critic returned invalid review lists.")

    critique = review.get("critique", "")

    if not isinstance(critique, str):
        raise ValueError("The critic explanation must be a string.")

    review_issues = (
        issues + unsupported + ambiguities + corrections
    )

    return {
        "is_grounded": review["is_grounded"],
        "critique": critique,
        "review_issues": review_issues,
    }
