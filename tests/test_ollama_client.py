
import pytest
import httpx

from app.llm.ollama_client import OllamaClient


def test_empty_prompt():
    client = OllamaClient()

    with pytest.raises(ValueError):
        client.generate("   ")


def test_ollama_response(monkeypatch):
    def mock_post(self, url, json):
        assert url.endswith("/api/chat")
        assert json["model"] == "qwen2.5:3b"

        request = httpx.Request("POST", url)

        return httpx.Response(
            status_code=200,
            request=request,
            json={
                "message": {
                    "role": "assistant",
                    "content": "RAG combines retrieval and generation.",
                }
            },
        )

    monkeypatch.setattr(
        httpx.Client,
        "post",
        mock_post,
    )

    client = OllamaClient()
    answer = client.generate("Explain RAG")

    assert answer == (
        "RAG combines retrieval and generation."
    )
