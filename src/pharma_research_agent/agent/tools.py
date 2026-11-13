
import json
from pathlib import Path
from typing import Any, Callable

from pharma_research_agent.config import CORPUS_DIR


# These schemas describe the tools to the language model.
# The model uses them to decide which local function to call.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Search the local pharmaceutical document corpus "
                "for relevant evidence using semantic similarity. "
                "Use this when you need to find information about "
                "medications, interactions, contraindications, "
                "side effects, or clinical recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused research query describing "
                            "the information to find."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Maximum number of relevant passages "
                            "to retrieve. Use a value between 1 and 10."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source_document",
            "description": (
                "Read the complete original Markdown document "
                "from the local corpus. Use this when you need "
                "additional context from a source document. "
                "The source must be a filename returned by search_corpus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "The source filename returned by search_corpus, "
                            "for example dailymed-warfarin.md."
                        ),
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
]


def create_tool_registry(vector_store) -> dict[str, Callable]:
    """
    Create the local tool registry.

    The vector store is injected into this function so that the
    corpus index can be built once and reused by all tool calls.
    """

    def search_corpus(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the indexed corpus for relevant passages."""

        if not query or not query.strip():
            return {"error": "The search query cannot be empty."}

        top_k = max(1, min(int(top_k), 10))

        results = vector_store.search(
            query=query.strip(),
            top_k=top_k,
        )

        return {
            "query": query,
            "results": results,
        }

    def read_source_document(source: str) -> dict[str, Any]:
        """Read an original Markdown document safely."""

        if not source or not source.strip():
            return {"error": "The source filename cannot be empty."}

        corpus_root = CORPUS_DIR.resolve()
        requested_path = (corpus_root / source).resolve()

        # Prevent access to files outside the corpus directory.
        if (
            requested_path != corpus_root
            and corpus_root not in requested_path.parents
        ):
            return {"error": "The requested source is not allowed."}

        if requested_path.suffix.lower() != ".md":
            return {"error": "Only Markdown documents can be read."}

        if not requested_path.is_file():
            return {"error": "The requested source document was not found."}

        content = requested_path.read_text(encoding="utf-8")

        return {
            "source": requested_path.relative_to(corpus_root).as_posix(),
            "content": content,
        }

    return {
        "search_corpus": search_corpus,
        "read_source_document": read_source_document,
    }


def execute_tool_call(
    tool_call,
    tool_registry: dict[str, Callable],
) -> str:
    """
    Execute a tool requested by the language model.

    Returns the tool result as a JSON string, ready to be
    sent back to the model.
    """

    tool_name = tool_call.function.name

    if tool_name not in tool_registry:
        return json.dumps(
            {"error": f"Unknown tool: {tool_name}"},
            ensure_ascii=False,
        )

    try:
        arguments = json.loads(tool_call.function.arguments)

        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")

        result = tool_registry[tool_name](**arguments)

        return json.dumps(result, ensure_ascii=False)

    except (json.JSONDecodeError, TypeError, ValueError):
        return json.dumps(
            {"error": "Invalid arguments supplied to the requested tool."},
            ensure_ascii=False,
        )
    except Exception:
        return json.dumps(
            {"error": "The requested local tool could not be executed."},
            ensure_ascii=False,
        )
