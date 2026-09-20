"""
Loads synthetic maintenance documents for the RAG pipeline.
Each document is returned with its content and basic metadata,
ready to be passed into the chunking stage.
"""

from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent.parent / "docs" / "synthetic"


@dataclass
class Document:
    source: str      # filename, used for citation
    content: str      # raw markdown text


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """
    Loads all .md files from the given directory.
    Raises FileNotFoundError if the directory doesn't exist or is empty,
    so a misconfigured path fails loudly instead of silently returning [].
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    documents = []
    for file_path in sorted(docs_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        documents.append(Document(source=file_path.name, content=content))

    if not documents:
        raise FileNotFoundError(f"No .md documents found in {docs_dir}")

    return documents