
import os

import httpx
import streamlit as st


# ==================================================
# 1. CONFIGURATION
# ==================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

ASK_ENDPOINT = f"{API_URL}/ask"
HEALTH_ENDPOINT = f"{API_URL}/health"


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Pharma Research Assistant",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Pharmaceutical Research Assistant")

st.write(
    """
    Your AI-powered pharmaceutical research assistant.

    Ask questions about medications, contraindications,
    interactions, and pharmaceutical information.
    """
)

st.caption(
    "Powered by FastAPI, LangGraph, RAG and OpenAI-compatible models."
)


# ==================================================
# 3. BACKEND CONNECTION
# ==================================================

def check_backend() -> bool:
    """Check whether the FastAPI backend is available."""

    try:
        response = httpx.get(
            HEALTH_ENDPOINT,
            timeout=5.0,
        )

        return response.status_code == 200

    except httpx.RequestError:
        return False


# ==================================================
# 4. QUESTION INPUT
# ==================================================

st.subheader("Ask your research question")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What are the main contraindications "
        "of warfarin?"
    ),
    height=120,
)

ask_button = st.button(
    "Research question",
    type="primary",
    disabled=not question.strip(),
)


# ==================================================
# 5. SEND QUESTION TO THE BACKEND
# ==================================================

if ask_button:

    with st.spinner(
        "The agent is planning, researching and reviewing..."
    ):

        try:
            response = httpx.post(
                ASK_ENDPOINT,
                json={
                    "question": question.strip()
                },
                timeout=300.0,
            )

            response.raise_for_status()

            result = response.json()

            st.success("Research completed successfully!")

            # ------------------------------------------
            # Display answer
            # ------------------------------------------

            st.subheader("Research answer")

            st.markdown(result["answer"])

            # ------------------------------------------
            # Display critic assessment
            # ------------------------------------------

            st.subheader("Grounding assessment")

            if result["grounded"]:
                st.success(
                    "The critic assessed the answer as grounded "
                    "in the retrieved evidence."
                )
            else:
                st.warning(
                    "The critic could not fully verify the answer."
                )

            # ------------------------------------------
            # Display source documents
            # ------------------------------------------

            st.subheader("Source documents")

            sources = result.get("sources", [])

            if sources:
                for source in sources:
                    st.markdown(f"- `{source}`")
            else:
                st.info("No source documents were returned.")

            # ------------------------------------------
            # Display research plan
            # ------------------------------------------

            with st.expander("View agent research plan"):

                for index, subquestion in enumerate(
                    result.get("subquestions", []),
                    start=1,
                ):
                    st.markdown(
                        f"{index}. {subquestion}"
                    )

        except httpx.HTTPStatusError as exc:

            st.error(
                "The backend returned an error. "
                f"HTTP status: {exc.response.status_code}"
            )

        except httpx.RequestError:

            st.error(
                "Could not connect to the backend. "
                "Make sure the FastAPI server is running."
            )

        except (KeyError, ValueError):

            st.error(
                "The backend returned an unexpected response."
            )


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.header("System status")

    if check_backend():
        st.success("Backend connected")
    else:
        st.error("Backend unavailable")

    st.divider()

    st.subheader("Technologies")

    st.markdown(
        """
        - Python
        - Streamlit
        - FastAPI
        - LangGraph
        - OpenAI-compatible API
        - NumPy
        """

    )

    st.divider()

    st.caption(
        "For research and educational purposes only. "
        "This application does not provide medical advice."
    )
