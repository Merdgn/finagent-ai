
import re

from app.rag.document_loader import Document
from app.rag.chunker import DocumentChunk


def chunk_markdown_by_heading(
    documents: list[Document],
) -> list[DocumentChunk]:
    """
    Split Markdown documents by heading sections.

    Each section preserves its heading and source.
    This first version assumes short sections.
    """

    chunks = []

    for document in documents:
        lines = document.content.splitlines()

        sections = []
        current_heading = "Introduction"
        current_lines = []

        for line in lines:
            heading_match = re.match(
                r"^#{1,6}\s+(.+)$",
                line.strip(),
            )

            if heading_match:
                if current_lines:
                    sections.append(
                        (
                            current_heading,
                            "\n".join(current_lines).strip(),
                        )
                    )

                current_heading = heading_match.group(1)
                current_lines = []

            else:
                if line.strip():
                    current_lines.append(line)

        if current_lines:
            sections.append(
                (
                    current_heading,
                    "\n".join(current_lines).strip(),
                )
            )

        for index, (heading, content) in enumerate(sections):
            if not content:
                continue

            chunk_content = (
                f"Document: {document.source}\n"
                f"Section: {heading}\n\n"
                f"{content}"
            )

            chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    source=document.source,
                    chunk_id=f"{document.source}:section:{index}",
                )
            )

    return chunks
