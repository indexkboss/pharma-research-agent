
import os
# reads environment variables, such as the model name and API configuration
from pathlib import Path
from pydoc import resolve
# handles file and directory paths in a cross-platform way
from dotenv import load_dotenv
# loads variables from the .env file
from openai import OpenAI
# creates the client that communicates with Syntax's OpenAI-compatible API


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
# __file__ is a special variable that holds the path of the current file
# resolve() returns the absolute path, 
# and parents[2] goes up two levels in the directory hierarchy to get the project root
# therefore BASE_DIR points to pharma-research-agent/
CORPUS_DIR = BASE_DIR / "corpus"
#CORPUS_DIR points to pharma-research-agent/corpus/


# Load environment variables from the project root
load_dotenv(BASE_DIR / ".env")

# LLM configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def validate_config() -> None:
    """Checks that the required configuration (variables and the directory) is available"""
    required_variables = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ]

    missing_variables = [
        name for name in required_variables
        if not os.getenv(name)
    ]

    if missing_variables:
        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing_variables)
        )

    if not CORPUS_DIR.is_dir():
        raise RuntimeError(
            f"Corpus directory not found: {CORPUS_DIR}"
        )


def create_client() -> OpenAI:
    """Create the OpenAI-compatible API client (after validating the configuration)"""
    validate_config()
    return OpenAI()
