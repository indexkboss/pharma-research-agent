
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pharma_research_agent.agent.workflow import create_workflow
from pharma_research_agent.retrieval.pipeline import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize resources once when the API starts.

    The corpus index and LangGraph agent are reused
    across incoming requests.
    """

    print("[API] Initializing pharmaceutical research agent...")

    try:
        vector_store, document_count, chunk_count = build_index()

        app.state.agent = create_workflow(vector_store)

        app.state.document_count = document_count
        app.state.chunk_count = chunk_count
        app.state.ready = True

        print("[API] Agent initialized successfully.")

    except Exception as exc:
        app.state.ready = False

        # Do not print exception details that could expose configuration.
        print(f"[API] Initialization failed: {type(exc).__name__}")
        raise

    yield

    print("[API] Shutting down.")


app = FastAPI(
    title="Pharmaceutical Research Agent",
    description=(
        "An agentic research assistant that answers pharmaceutical "
        "questions using a local document corpus and cited evidence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        min_length=1,
        max_length=5000,
        description="The pharmaceutical research question.",
    )


class AskResponse(BaseModel):
    """Structured response returned by the /ask endpoint."""

    answer: str
    sources: list[str]
    grounded: bool
    subquestions: list[str]


@app.get("/health")
def health():
    """Check whether the API and agent are ready."""

    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="The research agent is not ready.",
        )

    return {
        "status": "healthy",
        "agent_ready": True,
        "documents_indexed": app.state.document_count,
        "chunks_indexed": app.state.chunk_count,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Run the research agent for a user's question."""

    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="The research agent is not ready.",
        )

    try:
        result = app.state.agent.invoke(
            {
                "question": request.question.strip(),
                "revision_count": 0,
            }
        )

        sources = sorted({
            item["source"]
            for item in result.get("evidence", [])
            if item.get("source")
        })

        return AskResponse(
            answer=result["final_answer"],
            sources=sources,
            grounded=result.get("is_grounded", False),
            subquestions=result.get("subquestions", []),
        )

    except Exception:
        # Keep internal errors and configuration details out of the API response.
        raise HTTPException(
            status_code=500,
            detail="The research agent could not process this question.",
        )
