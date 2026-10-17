
from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore


def main():
    print("\nFinAgent AI - Semantic Search Demo\n")

    # Step 1: Load documents
    documents = load_documents("data/sample_docs")

    print(f"Loaded documents: {len(documents)}")

    # Step 2: Split documents into chunks
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100
    )

    print(f"Created chunks: {len(chunks)}")

    # Step 3: Generate embeddings
    embedding_service = EmbeddingService()

    texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.encode(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # Step 4: Build vector index
    vector_store = VectorStore()
    vector_store.build(chunks, embeddings)

    # Step 5: Search
    query = "What should I do if my credit card is lost?"

    query_embedding = embedding_service.encode([query])

    results = vector_store.search(
        query_embedding,
        top_k=3
    )

    print(f"\nUser question: {query}\n")

    for number, result in enumerate(results, start=1):
        print(f"--- Result {number} ---")
        print(f"Source: {result.source}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Similarity: {result.score:.4f}")
        print(f"Content:\n{result.content}\n")


if __name__ == "__main__":
    main()
