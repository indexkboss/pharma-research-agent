
from langgraph.graph import END, START, StateGraph

from pharma_research_agent.agent.answer import generate_draft_answer
from pharma_research_agent.agent.critic import review_answer
from pharma_research_agent.agent.planner import plan_question
from pharma_research_agent.agent.researcher import research_subquestions
from pharma_research_agent.agent.state import AgentState


MAX_REVISIONS = 1

MEDICAL_DISCLAIMER = (
    "\n\n---\n\n"
    "**Medical disclaimer:** This information is for research and "
    "educational purposes only. It is not medical advice and is not "
    "a substitute for consultation with a qualified healthcare "
    "professional."
)


def create_workflow(vector_store):
    """
    Create the complete pharmaceutical research agent.

    The vector store is injected so the corpus index is reused
    throughout the workflow.
    """

    # Node 1: Plan the user's question.
    def planner_node(state: AgentState) -> dict:
        print("\n[WORKFLOW] Planning the question...")
        return plan_question(state)

    # Node 2: Research using dynamically selected local tools.
    def researcher_node(state: AgentState) -> dict:
        print("\n[WORKFLOW] Researching the question...")

        revision_count = state.get("revision_count", 0)

        if revision_count == 0:
            questions_to_research = state.get("subquestions", [])
        else:
            # On revision, focus on the problems identified by the critic.
            issues = state.get("review_issues", [])

            issue_text = "; ".join(str(issue) for issue in issues[:5])

            questions_to_research = [
                (
                    "Find additional evidence in the local corpus to "
                    "address these evidence-review issues: "
                    f"{issue_text}. Original question: {state['question']}"
                )
            ]

        result = research_subquestions(
            question=state["question"],
            subquestions=questions_to_research,
            vector_store=vector_store,
        )

        # Preserve previous evidence and tool history during a revision.
        previous_evidence = state.get("evidence", [])
        previous_tool_calls = state.get("tool_calls", [])

        return {
            "evidence": previous_evidence + result["evidence"],
            "tool_calls": previous_tool_calls + result["tool_calls"],
        }

    # Node 3: Generate a draft using the available evidence.
    def answer_node(state: AgentState) -> dict:
        print("\n[WORKFLOW] Generating the draft answer...")
        return generate_draft_answer(state)

    # Node 4: Review the draft against the evidence.
    def critic_node(state: AgentState) -> dict:
        print("\n[WORKFLOW] Reviewing the draft...")
        result = review_answer(state)

        revision_count = state.get("revision_count", 0)

        if not result["is_grounded"]:
            revision_count += 1

        result["revision_count"] = revision_count

        return result

    # Node 5: Prepare the final user-facing response.
    def finalizer_node(state: AgentState) -> dict:
        print("\n[WORKFLOW] Preparing the final answer...")

        final_answer = state.get("draft_answer", "")

        if not state.get("is_grounded", False):
            issues = state.get("review_issues", [])

            limitation = (
                "\n\n**Evidence limitations:** The critic identified "
                "issues that could not be fully resolved using the "
                "available corpus."
            )

            if issues:
                limitation += "\n\nUnresolved points:\n" + "\n".join(
                    f"- {issue}" for issue in issues[:5]
                )

            final_answer += limitation

        final_answer += MEDICAL_DISCLAIMER

        return {"final_answer": final_answer}

    # Decide whether to retry research or finish.
    def route_after_critic(state: AgentState) -> str:
        if state.get("is_grounded", False):
            print("[WORKFLOW] Draft accepted by the critic.")
            return "finalize"

        if state.get("revision_count", 0) < MAX_REVISIONS + 1:
            print("[WORKFLOW] More evidence is needed. Retrying research.")
            return "researcher"

        print("[WORKFLOW] Revision limit reached. Finalizing with limitations.")
        return "finalize"

    # Build the LangGraph state machine.
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("answer", answer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "answer")
    graph.add_edge("answer", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",
            "finalize": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()
