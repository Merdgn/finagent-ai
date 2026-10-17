
from app.llm.ollama_client import OllamaClient


def main():
    print("\nFinAgent AI - Ollama Integration Test\n")

    client = OllamaClient()

    system_prompt = (
        "You are a helpful AI assistant. "
        "Answer clearly and concisely. "
        "Do not invent facts."
    )

    question = (
        "What is Retrieval-Augmented Generation? "
        "Explain it in two sentences."
    )

    print(f"Question: {question}\n")

    answer = client.generate(
        prompt=question,
        system_prompt=system_prompt,
    )

    print("LLM Response:")
    print(answer)


if __name__ == "__main__":
    main()
