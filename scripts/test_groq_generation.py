
from dotenv import load_dotenv

from app.llm.groq_client import GroqClient


def main():
    load_dotenv()

    client = GroqClient()

    questions = [
        "Say hello in Turkish in one sentence.",
        "Explain RAG in one sentence.",
    ]

    for question in questions:
        print("\nQUESTION:", question)

        response = client.generate(
            prompt=question,
            system_prompt=(
                "You are FinAgent AI. "
                "Answer concisely in the requested language."
            ),
        )

        print("ANSWER:", response)


if __name__ == "__main__":
    main()
