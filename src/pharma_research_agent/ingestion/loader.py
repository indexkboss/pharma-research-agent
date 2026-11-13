'''
loader.py is responsible for loading all documents from the corpus 
and preparing them for the rest of the ingestion pipeline

It should preserve two things:
- The document's content 
- Its original filename

Why the filename?
Because when the LLM answers a question, we need to know the source of the retrieved information

this code :
        1. Check whether the corpus directory exists.
        2. Find all .md files inside it, including files in subdirectories.
        3. Read the content of each file.
        4. Ignore empty files.
        5. Keep the file's relative path and its content together.
        6. Return a list of documents.
'''


from dataclasses import dataclass #to create classes designed mainly to store data
from pathlib import Path

from pharma_research_agent.config import CORPUS_DIR


@dataclass(frozen=True)
# this decorator autoamatically generated constructor and otheruseful methods
# frozen=True: 
# the original documents are treated as read-only data structures it prevents accidental modification
class SourceDocument:
    """Represents one original document from the corpus."""

    source: str
    content: str


def load_markdown_documents(
    corpus_dir: Path = CORPUS_DIR,
) -> list[SourceDocument]:
    """
    Load all Markdown documents from the corpus directory.

    Each document keeps its relative source path and content.
    """

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}"
        )

    paths = sorted(corpus_dir.rglob("*.md"))#Searches recursively through subdirectories

    if not paths:
        raise ValueError(
            f"No Markdown documents found in: {corpus_dir}"
        )

    documents = []

    for path in paths:
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        source = path.relative_to(corpus_dir).as_posix()#It removes the common parent directory

        #create a SourceDocument instance 
        document = SourceDocument(
            source=source,
            content=content,
        )

        documents.append(document)

    if not documents:
        raise ValueError("All Markdown documents are empty.")

    return documents
