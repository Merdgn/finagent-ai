
from app.agents.workflow import create_agent_workflow
from app.rag.rag_pipeline import RAGResponse


class FakeRAGPipeline:
    def ask(self, question: str):
        return RAGResponse(
            answer="Report the lost card through official channels.",
            sources=["credit_card.md"],
            retrieved_chunks=1,
        )


def test_agent_rag_route():
    agent = create_agent_workflow(FakeRAGPipeline())

    result = agent.invoke({
        "question": "My credit card is lost"
    })

    assert result["route"] == "rag"
    assert result["sources"] == ["credit_card.md"]


def test_agent_calculator():
    agent = create_agent_workflow(FakeRAGPipeline())

    result = agent.invoke({
        "question": "calculate: 250 * 1.15"
    })

    assert result["route"] == "calculator"
    assert result["answer"] == "Result: 287.50"


def test_agent_safety_route():
    agent = create_agent_workflow(FakeRAGPipeline())

    result = agent.invoke({
        "question": "What is the weather in Tokyo?"
    })

    assert result["route"] == "safety"
    assert result["sources"] == []


def test_agent_invalid_calculation():
    agent = create_agent_workflow(FakeRAGPipeline())

    result = agent.invoke({
        "question": "calculate: 10 / 0"
    })

    assert result["route"] == "calculator"
    assert result["answer"] == "Invalid arithmetic operation."
