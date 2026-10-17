
import json
import time
from pathlib import Path

from app.rag.document_loader import load_documents
from app.rag.semantic_chunker import chunk_markdown_by_heading
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore


DATASET = Path("data/evaluation/retrieval_cases.json")
TOP_K = 3
THRESHOLD = 0.30


def evaluate():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    documents = load_documents("data/sample_docs")
    chunks = chunk_markdown_by_heading(documents)

    embedder = EmbeddingService()
    vectors = embedder.encode([c.content for c in chunks])

    store = VectorStore()
    store.build(chunks, vectors)

    print("\nFINAGENT AI - RETRIEVAL EVALUATION")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Top K: {TOP_K}")
    print(f"Threshold: {THRESHOLD}\n")

    hits = []
    reciprocal_ranks = []
    latencies = []
    ood_results = []

    for case in cases:
        question = case["question"]

        start = time.perf_counter()

        query_vector = embedder.encode([question])

        results = store.search(
            query_vector,
            top_k=TOP_K,
            min_score=THRESHOLD,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        expected_source = case["expected_source"]
        expected_section = case["expected_section"]

        print(f"\n{case['id']}: {question}")
        print(f"Latency: {latency_ms:.2f} ms")

        for rank, result in enumerate(results, start=1):
            print(
                f"  Rank {rank} | "
                f"Score {result.score:.4f} | "
                f"{result.chunk_id}"
            )

        if expected_source is None:
            rejected = len(results) == 0
            ood_results.append(rejected)
            print(f"Out-of-domain rejected: {rejected}")
            continue

        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            is_relevant = (
                result.source == expected_source
                and f"Section: {expected_section}\n"
                in result.content
            )

            if is_relevant:
                first_relevant_rank = rank
                break

        hit = first_relevant_rank is not None

        hits.append(int(hit))
        reciprocal_ranks.append(
            1 / first_relevant_rank if hit else 0.0
        )

        print(f"Hit@{TOP_K}: {hit}")
        print(f"First relevant rank: {first_relevant_rank}")

    hit_rate = sum(hits) / len(hits) if hits else 0.0
    mrr = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Hit Rate@{TOP_K}: {hit_rate:.2%}")
    print(f"MRR@{TOP_K}: {mrr:.4f}")
    print(f"Average retrieval latency: {avg_latency:.2f} ms")

    if ood_results:
        ood_rejection_rate = (
            sum(ood_results) / len(ood_results)
        )
        print(
            "Out-of-domain rejection rate: "
            f"{ood_rejection_rate:.2%}"
        )


if __name__ == "__main__":
    evaluate()
