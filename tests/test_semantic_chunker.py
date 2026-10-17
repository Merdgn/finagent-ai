
from app.rag.document_loader import Document
from app.rag.semantic_chunker import (
    chunk_markdown_by_heading,
)


def test_markdown_section_chunking():
    document = Document(
        content=(
            "# Credit Card\n\n"
            "Product information.\n\n"
            "## Lost Card\n\n"
            "Report a lost card immediately.\n\n"
            "## Security\n\n"
            "Never share your PIN."
        ),
        source="credit_card.md",
    )

    chunks = chunk_markdown_by_heading([document])

    assert len(chunks) == 3

    lost_card_chunks = [
        chunk
        for chunk in chunks
        if "Section: Lost Card" in chunk.content
    ]

    assert len(lost_card_chunks) == 1
    assert "Report a lost card immediately" in (
        lost_card_chunks[0].content
    )


def test_empty_document():
    document = Document(
        content="",
        source="empty.md",
    )

    chunks = chunk_markdown_by_heading([document])

    assert chunks == []
