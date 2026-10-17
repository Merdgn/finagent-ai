
from dataclasses import dataclass

import faiss
import numpy as np

from app.rag.chunker import DocumentChunk


@dataclass
class SearchResult:
    content: str
    source: str
    chunk_id: str
    score: float


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: list[DocumentChunk] = []

    def build(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray
    ) -> None:
        vectors = np.ascontiguousarray(
            embeddings, dtype=np.float32
        )

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        if not chunks or len(chunks) != len(vectors):
            raise ValueError(
                "Chunks and embeddings must match"
            )

        dimension = vectors.shape[1]

        if dimension == 0:
            raise ValueError("Invalid embedding dimension")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values")

        # Rebuild the index from scratch.
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        self.index = index
        self.chunks = list(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> list[SearchResult]:
        if self.index is None:
            raise RuntimeError("Index has not been built")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = np.asarray(
            query_embedding, dtype=np.float32
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if (
            query.ndim != 2
            or query.shape != (1, self.index.d)
        ):
            raise ValueError("Invalid query embedding shape")

        if not np.isfinite(query).all():
            raise ValueError("Invalid query values")

        query = np.ascontiguousarray(query)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]

            results.append(
                SearchResult(
                    content=chunk.content,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    score=float(score)
                )
            )

        return results
