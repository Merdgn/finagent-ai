
from fastapi.testclient import TestClient

from app.main import app, get_agent_workflow


client = TestClient(app)


class FakeAgent:
    def invoke(self, state):
        question = state["question"]

        if question.startswith("calculate:"):
            return {
                "answer": "Result: 287.50",
                "route": "calculator",
                "sources": [],
            }

        if "credit card" in question.lower():
            return {
                "answer": "Report the lost card.",
                "route": "rag",
                "sources": ["credit_card.md"],
            }

        return {
            "answer": "Unsupported question.",
            "route": "safety",
            "sources": [],
        }


def setup_function():
    app.dependency_overrides[get_agent_workflow] = (
        lambda: FakeAgent()
    )


def teardown_function():
    app.dependency_overrides.clear()


def test_agent_api_rag():
    response = client.post(
        "/agent/chat",
        json={
            "question": "My credit card is lost"
        },
    )

    assert response.status_code == 200
    assert response.json()["route"] == "rag"
    assert response.json()["sources"] == [
        "credit_card.md"
    ]


def test_agent_api_calculator():
    response = client.post(
        "/agent/chat",
        json={
            "question": "calculate: 250 * 1.15"
        },
    )

    assert response.status_code == 200
    assert response.json()["route"] == "calculator"
    assert response.json()["answer"] == "Result: 287.50"


def test_agent_api_empty_question():
    response = client.post(
        "/agent/chat",
        json={
            "question": "   "
        },
    )

    assert response.status_code == 422


def test_agent_api_safety():
    response = client.post(
        "/agent/chat",
        json={
            "question": "What is the weather in Tokyo?"
        },
    )

    assert response.status_code == 200
    assert response.json()["route"] == "safety"
