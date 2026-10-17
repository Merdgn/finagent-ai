
from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore


QUESTIONS = [
    "What should I do if my credit card is lost?",
    "How can I open a savings account?",
    "What is the exact interest rate for this account?",
    "What is the weather in Tokyo?",
]


def main():
    print("\nFinAgent AI - Retrieval Diagnostics\n")

    documents = load_documents("data/sample_docs")

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    embedding_service = EmbeddingService()

    embeddings = embedding_service.encode(
        [chunk.content for chunk in chunks]
    )

    store = VectorStore()
    store.build(chunks, embeddings)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    for question in QUESTIONS:
        print("\n" + "=" * 65)
        print(f"QUESTION: {question}")
        print("=" * 65)

        query_embedding = embedding_service.encode(
            [question]
        )

        # No filtering: inspect raw retrieval scores.
        results = store.search(
            query_embedding,
            top_k=5,
            min_score=None,
        )

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Score: {result.score:.4f}")
            print(f"Source: {result.source}")
            print(f"Chunk: {result.chunk_id}")

            preview = " ".join(result.content.split())
            print(f"Preview: {preview[:180]}")


if __name__ == "__main__":
    main()
