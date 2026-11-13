
from pharma_research_agent.agent.state import AgentState
from pharma_research_agent.config import OPENAI_MODEL, create_client


ANSWER_SYSTEM_PROMPT = """
You are the answer-generation component of a pharmaceutical
research assistant.

Your task is to answer the user's question using ONLY the
evidence provided from the local pharmaceutical corpus.

CORE RULES:

1. Base every medical or clinical claim on the supplied evidence.
2. Do not invent facts, recommendations, contraindications,
   statistics, mechanisms, or clinical guidelines.
3. If the evidence does not answer part of the question,
   explicitly state that the available corpus does not establish it.
4. Distinguish information from different source documents.
5. Preserve important clinical conditions, thresholds,
   and qualifications exactly as supported by the sources.
6. Cite claims using the source filename in square brackets.
   Example: [dailymed-warfarin.md]
7. Do not cite a source unless its evidence was supplied.
8. Explain uncertainty and conflicting evidence when present.
9. Do not present the response as a diagnosis or prescription.
10. Write a clear, structured answer.

IMPORTANT: DISTINGUISH THREE TYPES OF INFORMATION.

A. Explicit instructions and recommendations:
   - Only describe something as a label recommendation,
     instruction, or requirement if the supplied source
     explicitly states it.
   - Do not turn a general warning into a specific monitoring
     protocol that the label does not provide.
   - Do not claim that a label instructs patients to report
     particular symptoms unless that instruction appears
     in the supplied evidence.

B. Reported risks and adverse events:
   - Distinguish warnings, clinical-study findings,
     and postmarketing reports.
   - When discussing voluntary postmarketing reports,
     preserve the source's limitations.
   - If the source says frequency or causality cannot be
     reliably established, explicitly include that qualification.
   - Do not present a reported event as proof of causation.

C. Clinical considerations and inferences:
   - Do not present a reasonable clinical consideration
     as an explicit recommendation from a drug label.
   - If a consideration is not explicitly stated in the evidence,
     omit it unless it is necessary to explain an evidence limitation.
   - If included, clearly identify it as an inference rather than
     a source-supported instruction.

SPECIAL CARE WITH DRUG INTERACTIONS:

When discussing a potential interaction:
- State the interaction risk only to the extent supported
  by the supplied sources.
- Preserve the exact monitoring language used by the source.
- Do not invent a monitoring frequency or schedule.
- Do not infer a pharmacological mechanism unless the evidence
  explicitly supports it.
- If a source reports that an in-vitro study did not demonstrate
  an expected interaction mechanism, preserve that qualification.
- Distinguish the existence of a warning from evidence about
  the frequency or magnitude of clinical harm.

The response must include:
- A direct answer to the question.
- Relevant evidence and explanations.
- Explicitly supported monitoring or management instructions.
- Important limitations in the available evidence.

If the evidence does not specify a monitoring schedule,
say so rather than proposing one.

Do not include a medical disclaimer in this draft.
The final workflow will add the required disclaimer.
"""


def format_evidence(evidence: list[dict]) -> str:
    """Format and limit retrieved evidence for the language model."""

    formatted_items = []
    seen_items = set()

    for index, item in enumerate(evidence, start=1):
        source = item.get("source", "Unknown source")
        text = item.get("text", "")

        if not text:
            continue

        # Avoid sending identical evidence more than once.
        fingerprint = (source, text[:500])

        if fingerprint in seen_items:
            continue

        seen_items.add(fingerprint)

        # Limit the size of each evidence item.
        text = text[:3500]

        formatted_items.append(
            f"[Evidence {index}]\n"
            f"Source: {source}\n"
            f"Type: {item.get('kind', 'retrieved_passage')}\n"
            f"Content:\n{text}"
        )

        # Keep the overall context manageable.
        if len("\n\n".join(formatted_items)) >= 24000:
            break

    if not formatted_items:
        return "No usable evidence was retrieved."

    return "\n\n---\n\n".join(formatted_items)


def generate_draft_answer(state: AgentState) -> dict:
    """Generate a preliminary evidence-based answer."""

    question = state["question"]
    evidence = state.get("evidence", [])

    if not evidence:
        return {
            "draft_answer": (
                "The available local corpus did not provide "
                "sufficient evidence to answer this question."
            )
        }

    formatted_evidence = format_evidence(evidence)

    messages = [
        {
            "role": "system",
            "content": ANSWER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Research plan:\n{state.get('subquestions', [])}\n\n"
                f"Evidence:\n{formatted_evidence}\n\n"
                "Write a preliminary answer using only this evidence."
            ),
        },
    ]

    client = create_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1800,
    )

    draft = response.choices[0].message.content

    if not draft:
        raise RuntimeError("The answer generator returned an empty response.")

    return {"draft_answer": draft.strip()}
