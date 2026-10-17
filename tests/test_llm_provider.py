
from unittest.mock import patch

import pytest

from app.llm.provider import get_llm_client


def test_groq_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv(
        "GROQ_MODEL", "openai/gpt-oss-20b"
    )

    with patch(
        "app.llm.provider.GroqClient"
    ) as mock_client:
        get_llm_client()
        mock_client.assert_called_once_with(
            model="openai/gpt-oss-20b"
        )


def test_ollama_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    with patch(
        "app.llm.provider.OllamaClient"
    ) as mock_client:
        get_llm_client()
        mock_client.assert_called_once()


def test_invalid_provider(monkeypatch):
    monkeypatch.setenv(
        "LLM_PROVIDER", "invalid-provider"
    )

    with pytest.raises(ValueError):
        get_llm_client()
