
import numpy as np

from app.rag.rag_pipeline import RAGPipeline
from app.rag.vector_store import SearchResult


class FakeEmbeddingService:
    def encode(self, texts):
        return np.ones(
            (len(texts), 3),
            dtype=np.float32,
        )


class FakeVectorStore:
    def build(self, chunks, embeddings):
        self.chunks = chunks

    def search(self, query_embedding, top_k=3):
        return [
            SearchResult(
                content=(
                    "A lost card should be reported "
                    "through official bank channels."
                ),
                source="credit_card.md",
                chunk_id="credit_card.md:0",
                score=0.95,
            )
        ]


class FakeLLMClient:
    def generate(self, prompt, system_prompt=None):
        assert "official bank channels" in prompt
        assert system_prompt is not None

        return "Report the lost card immediately."


def test_rag_pipeline_answer():
    pipeline = RAGPipeline(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        llm_client=FakeLLMClient(),
    )

    result = pipeline.ask(
        "What should I do if I lose my card?"
    )

    assert result.answer == (
        "Report the lost card immediately."
    )
    assert result.sources == ["credit_card.md"]
    assert result.retrieved_chunks == 1


def test_rag_pipeline_empty_question():
    pipeline = RAGPipeline(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        llm_client=FakeLLMClient(),
    )

    try:
        pipeline.ask("   ")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for empty question"
        )
