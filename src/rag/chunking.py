"""
Splits loaded documents into retrievable chunks.
Chunking strategy: split by markdown ## sections rather than fixed
character windows, since cutting a threshold explanation mid-sentence
would hurt retrieval quality on these structured documents.
"""

import re
from dataclasses import dataclass

from src.rag.document_loader import Document

MAX_CHUNK_CHARS = 800  # sections longer than this get split further


@dataclass
class Chunk:
    source: str        # originating filename, for citations
    section: str        # section heading (e.g. "Detection thresholds")
    content: str         # chunk text


def chunk_document(document: Document) -> list[Chunk]:
    """
    Splits a single document into chunks along ## headings.
    The top-level # title is not a chunk on its own: it is prepended
    to every section chunk so each one keeps the failure mode name.
    """
    # Split on lines starting with "## " (level-2 headings)
    parts = re.split(r"\n(?=## )", document.content.strip())

    # First part holds the "# Title" line (and any intro text before the first ##)
    title_match = re.match(r"^# (.+)", parts[0])
    doc_title = title_match.group(1).strip() if title_match else document.source
    if title_match:
        intro = parts[0].split("\n", 1)[1].strip() if "\n" in parts[0] else ""
        parts = ([intro] if intro else []) + parts[1:]

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract section title if present
        heading_match = re.match(r"^#{1,2}\s+(.+)", part)
        section_title = heading_match.group(1) if heading_match else "Introduction"

        # Fallback: split oversized sections into smaller pieces
        pieces = [part] if len(part) <= MAX_CHUNK_CHARS else _split_long_section(part)
        for piece in pieces:
            chunks.append(
                Chunk(
                    source=document.source,
                    section=section_title,
                    content=f"# {doc_title}\n\n{piece}",
                )
            )

    return chunks


def _split_long_section(text: str) -> list[str]:
    """Splits a long section into ~MAX_CHUNK_CHARS pieces on paragraph boundaries."""
    paragraphs = text.split("\n\n")
    pieces, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= MAX_CHUNK_CHARS:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                pieces.append(current)
            current = para

    if current:
        pieces.append(current)

    return pieces


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunks a list of documents into a flat list of chunks."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks