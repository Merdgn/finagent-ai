
import numpy as np
import pytest

from app.rag.chunker import DocumentChunk
from app.rag.vector_store import VectorStore


def test_vector_store_search():
    chunks = [
        DocumentChunk(
            content="Lost credit card information",
            source="credit_card.md",
            chunk_id="credit_card.md:0"
        ),
        DocumentChunk(
            content="Savings account information",
            source="savings_account.md",
            chunk_id="savings_account.md:0"
        ),
    ]

    # Synthetic normalized embeddings
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32
    )

    store = VectorStore()
    store.build(chunks, embeddings)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0].source == "credit_card.md"
    assert results[0].score == pytest.approx(1.0)


def test_search_before_build():
    store = VectorStore()

    with pytest.raises(RuntimeError):
        store.search(
            np.array([1.0, 0.0], dtype=np.float32)
        )


def test_invalid_embedding_count():
    store = VectorStore()

    chunks = [
        DocumentChunk(
            content="Example",
            source="test.md",
            chunk_id="test.md:0"
        )
    ]

    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32
    )

    with pytest.raises(ValueError):
        store.build(chunks, embeddings)



def test_similarity_threshold():
    chunks = [
        DocumentChunk(
            content="Lost credit card instructions",
            source="credit_card.md",
            chunk_id="credit_card.md:0",
        ),
        DocumentChunk(
            content="Savings account information",
            source="savings_account.md",
            chunk_id="savings_account.md:0",
        ),
    ]

    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    store = VectorStore()
    store.build(chunks, embeddings)

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = store.search(
        query,
        top_k=2,
        min_score=0.5,
    )

    assert len(results) == 1
    assert results[0].source == "credit_card.md"


def test_no_results_above_threshold():
    chunks = [
        DocumentChunk(
            content="Savings account",
            source="savings_account.md",
            chunk_id="savings_account.md:0",
        )
    ]

    embeddings = np.array(
        [[0.0, 1.0]],
        dtype=np.float32,
    )

    store = VectorStore()
    store.build(chunks, embeddings)

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = store.search(
        query,
        top_k=1,
        min_score=0.5,
    )

    assert results == []
