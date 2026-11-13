
import pytest

from fastapi.testclient import TestClient

from pharma_research_agent.api import app


class FakeAgent:
    """Simulate the LangGraph agent without calling an LLM."""

    def invoke(self, state):
        return {
            "question": state["question"],
            "final_answer": (
                "Warfarin is contraindicated in certain "
                "clinical situations. [dailymed-warfarin.md]"
            ),
            "is_grounded": True,
            "subquestions": [
                "Identify the main contraindications of warfarin."
            ],
            "evidence": [
                {
                    "source": "dailymed-warfarin.md",
                    "text": "Example evidence.",
                }
            ],
        }


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked initialization."""

    def fake_build_index():
        return object(), 21, 467

    def fake_create_workflow(vector_store):
        return FakeAgent()

    monkeypatch.setattr(
        "pharma_research_agent.api.build_index",
        fake_build_index,
    )

    monkeypatch.setattr(
        "pharma_research_agent.api.create_workflow",
        fake_create_workflow,
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """The health endpoint should report a ready agent."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["agent_ready"] is True
    assert data["documents_indexed"] == 21
    assert data["chunks_indexed"] == 467


def test_ask_endpoint(client):
    """The ask endpoint should return a structured answer."""

    response = client.post(
        "/ask",
        json={
            "question": "What are the main contraindications of warfarin?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert "grounded" in data
    assert "subquestions" in data

    assert data["grounded"] is True
    assert "dailymed-warfarin.md" in data["sources"]


def test_ask_rejects_empty_question(client):
    """The API should reject an empty question."""

    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_rejects_missing_question(client):
    """The API should reject a request without a question."""

    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422
