
from app.rag.rag_pipeline import RAGPipeline
from app.agents.workflow import create_agent_workflow


def main():
    print("\nFinAgent AI - LangGraph Agent Demo\n")

    rag_pipeline = RAGPipeline()

    agent = create_agent_workflow(rag_pipeline)

    questions = [
        "What should I do if my credit card is lost?",
        "calculate: 250 * 1.15",
        "What is the weather in Tokyo?",
    ]

    for question in questions:
        print("=" * 55)
        print(f"QUESTION: {question}")

        result = agent.invoke({"question": question})

        print(f"ROUTE: {result['route']}")
        print(f"ANSWER: {result['answer']}")
        print(f"SOURCES: {result.get('sources', [])}")
        print()


if __name__ == "__main__":
    main()
