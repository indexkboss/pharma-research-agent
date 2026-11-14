# Pharma Research Agent

An agentic pharmaceutical research assistant built with Python, LangGraph, Retrieval-Augmented Generation (RAG), and an OpenAI-compatible API.

The application answers pharmaceutical research questions using a local corpus of trusted drug information documents. It follows a multi-step agentic workflow to plan research, retrieve evidence, select tools, review generated answers, and produce source-grounded responses.

> **Medical disclaimer:** This application is intended for research and educational purposes only. It does not provide medical advice, diagnosis, or treatment recommendations. Always consult a qualified healthcare professional for medical decisions.

## 1. Project Overview

The objective of this project is to demonstrate how agentic AI can support pharmaceutical information research while prioritizing evidence grounding and transparency.

Unlike a basic question-answering chatbot, the agent follows a structured workflow:

1. **Plan:** Decompose a complex question into smaller research subquestions.
2. **Retrieve:** Search a local document corpus using semantic similarity.
3. **Use Tools:** Dynamically select and execute research tools to gather relevant evidence.
4. **Review:** Evaluate the draft answer for grounding, source accuracy, unsupported claims, and clinical ambiguity.
5. **Answer:** Generate a final response with source references and a medical disclaimer.

## 2. Architecture

The application uses the following components:

| Component | Technology | Responsibility |
|---|---|---|
| Agent orchestration | LangGraph | Controls the multi-step workflow |
| Language model | OpenAI-compatible API | Planning, tool selection, answer generation, and review |
| Embeddings | OpenAI-compatible embeddings API | Converts text into vector representations |
| Retrieval | NumPy | In-memory cosine similarity search |
| API | FastAPI | Exposes the research agent |
| Frontend | Streamlit | Provides a simple user interface |
| Data validation | Pydantic | Validates API requests and responses |
| Dependency management | uv | Manages Python dependencies and lockfile |

### Agent Workflow

The agent follows this workflow:

`Question → Planner → Researcher → Answer Generator → Critic → Final Answer`

If the critic identifies grounding problems, the workflow can trigger a revision before returning the final response.

### Local Research Tools

The agent has access to two local tools:

- `search_corpus`: Searches the indexed document chunks and retrieves relevant passages.
- `read_source_document`: Reads the original Markdown source document for additional context.

The language model selects tools dynamically through function calling.

## 3. Project Structure

```text
pharma-research-agent/
├── corpus/                 # Local pharmaceutical knowledge base
├── frontend/
│   └── app.py              # Streamlit frontend
├── reports/
│   └── evaluation_results.json
├── scripts/                # Development and evaluation scripts
├── src/
│   └── pharma_research_agent/
│       ├── agent/          # Planning, research, tools, critic, workflow
│       ├── ingestion/      # Document loading and chunking
│       ├── retrieval/      # Embeddings and vector search
│       ├── api.py          # FastAPI application
│       └── config.py       # Environment and API configuration
├── tests/
│   └── test_api.py         # API tests
├── .env.example            # Environment configuration template
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## 4. Requirements

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible API endpoint
- API credentials for the configured language model and embedding model

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/indexkboss/pharma-research-agent.git
cd pharma-research-agent
```

Install the project dependencies:

```bash
uv sync --locked
```

This command installs the dependencies using the versions recorded in `uv.lock`.

## 6. Configuration

Create a local `.env` file from the provided template:

```bash
# Windows CMD
copy .env.example .env
```

Configure the following variables in `.env`:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_api_base_url
OPENAI_MODEL=your_chat_model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Replace the placeholder values with credentials and model names supplied by your API provider.

**Never commit your `.env` file or expose API credentials publicly.**

## 7. Running the Application

### Start the FastAPI backend

```bash
uv run uvicorn pharma_research_agent.api:app --host 127.0.0.1 --port 8000
```

The API documentation is available at:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### Start the Streamlit frontend

Open another terminal:

```bash
uv run streamlit run frontend/app.py
```

The frontend provides an interface to submit pharmaceutical research questions and display the generated answers.

## 8. API Usage

### Health Check

**Endpoint:** `GET /health`

Example:

```bash
curl http://127.0.0.1:8000/health
```

### Ask a Research Question

**Endpoint:** `POST /ask`

Example request:

```json
{
  "question": "What are the main contraindications of warfarin?"
}
```

Example using curl:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are the main contraindications of warfarin?\"}"
```

The response includes the generated answer, source references, grounding status, and research subquestions.

## 9. Evaluation and Testing

Run the API tests:

```bash
uv run pytest tests/test_api.py -v
```

Run the official question evaluation script:

```bash
uv run python scripts/evaluate_questions.py
```

Additional scripts test the planner, retrieval tools, researcher, critic, complete workflow, and adversarial or edge-case scenarios.

Evaluation focuses on:

- Evidence grounding
- Source attribution
- Detection of unsupported claims
- Clinical ambiguity
- API behavior and input validation

The evaluation results are stored in `reports/evaluation_results.json`.

A positive grounding assessment is an automated model-based review, not independent clinical validation.

## 10. Limitations

- Answers depend on the content and coverage of the local document corpus.
- Retrieval quality depends on document chunking and embedding similarity.
- The critic is an automated language-model reviewer and cannot guarantee medical correctness.
- The application is not a substitute for professional clinical judgment.
- Model and embedding availability depend on the configured API provider.

## 11. Future Improvements

Potential improvements include:

- Expanding the pharmaceutical knowledge base.
- Adding stronger automated evaluation datasets.
- Improving citation-level verification.
- Adding authentication and production deployment configuration.
- Improving the frontend with richer evidence visualization.

## 12. License and Intended Use

This project was developed as a technical demonstration for a Generative AI Bootcamp coding assignment.

It is intended for educational and research purposes, not clinical decision-making.
