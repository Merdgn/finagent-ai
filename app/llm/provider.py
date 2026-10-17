
import os

from dotenv import load_dotenv

from app.llm.ollama_client import OllamaClient
from app.llm.groq_client import GroqClient


def get_llm_client():
    load_dotenv()

    provider = os.getenv(
        "LLM_PROVIDER", "ollama"
    ).strip().lower()

    if provider == "groq":
        return GroqClient(
            model=os.getenv(
                "GROQ_MODEL", "openai/gpt-oss-20b"
            )
        )

    if provider == "ollama":
        return OllamaClient()

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )
