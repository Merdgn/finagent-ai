
import pytest

from app.rag.document_loader import (
    Document,
    load_documents,
)
from app.rag.chunker import chunk_documents


def test_load_documents():
    documents = load_documents("data/sample_docs")

    assert len(documents) == 2

    sources = {doc.source for doc in documents}

    assert "savings_account.md" in sources
    assert "credit_card.md" in sources


def test_chunk_documents():
    documents = [
        Document(
            content="A" * 1200,
            source="sample.txt"
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100
    )

    assert len(chunks) == 3
    assert chunks[0].source == "sample.txt"
    assert all(
        len(chunk.content) <= 500
        for chunk in chunks
    )


def test_invalid_overlap():
    documents = [
        Document(content="Test", source="test.txt")
    ]

    with pytest.raises(ValueError):
        chunk_documents(
            documents,
            chunk_size=100,
            overlap=100
        )
