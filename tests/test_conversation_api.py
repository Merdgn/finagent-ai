
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app, get_conversation_service
from app.memory.conversation_service import ConversationResponse


class FakeConversationService:
    def ask(self, session_id: str, question: str):
        return ConversationResponse(
            answer="Identity verification is required.",
            route="rag",
            sources=["savings_account.md"],
            standalone_question=question,
            session_id=session_id,
        )


def test_conversation_api_success():
    app.dependency_overrides[get_conversation_service] = (
        lambda: FakeConversationService()
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/agent/conversation",
                json={
                    "session_id": str(uuid4()),
                    "question": "How can I open a savings account?",
                },
            )

        assert response.status_code == 200

        data = response.json()

        assert data["route"] == "rag"
        assert data["sources"] == ["savings_account.md"]
        assert data["answer"] == (
            "Identity verification is required."
        )

    finally:
        app.dependency_overrides.clear()


def test_invalid_session_id():
    with TestClient(app) as client:
        response = client.post(
            "/agent/conversation",
            json={
                "session_id": "invalid-session",
                "question": "Hello",
            },
        )

    assert response.status_code == 422


def test_empty_question():
    with TestClient(app) as client:
        response = client.post(
            "/agent/conversation",
            json={
                "session_id": str(uuid4()),
                "question": "",
            },
        )

    assert response.status_code == 422


def test_missing_session_id():
    with TestClient(app) as client:
        response = client.post(
            "/agent/conversation",
            json={
                "question": "Hello",
            },
        )

    assert response.status_code == 422
