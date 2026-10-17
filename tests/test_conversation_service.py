
from uuid import uuid4

import pytest

from app.memory.conversation_store import ConversationStore
from app.memory.conversation_service import ConversationService


class FakeLLM:
    def generate(self, prompt, system_prompt=None):
        return (
            "What is the interest rate "
            "of the Demo Savings Account?"
        )


class FakeAgent:
    def __init__(self):
        self.questions = []

    def invoke(self, state):
        self.questions.append(state["question"])

        return {
            "answer": "Refer to the official disclosure.",
            "route": "rag",
            "sources": ["savings_account.md"],
        }


def test_first_message_without_history():
    store = ConversationStore()
    agent = FakeAgent()

    service = ConversationService(
        store=store,
        agent=agent,
        llm_client=FakeLLM(),
    )

    session_id = str(uuid4())

    result = service.ask(
        session_id,
        "How can I open a savings account?",
    )

    assert result.standalone_question == (
        "How can I open a savings account?"
    )

    assert len(store.get_history(session_id)) == 2


def test_follow_up_question_rewriting():
    store = ConversationStore()
    agent = FakeAgent()

    service = ConversationService(
        store=store,
        agent=agent,
        llm_client=FakeLLM(),
    )

    session_id = str(uuid4())

    service.ask(
        session_id,
        "How can I open a savings account?",
    )

    result = service.ask(
        session_id,
        "What about its interest rate?",
    )

    assert result.standalone_question == (
        "What is the interest rate "
        "of the Demo Savings Account?"
    )

    assert agent.questions[-1] == (
        result.standalone_question
    )

    assert len(store.get_history(session_id)) == 4


def test_session_isolation():
    store = ConversationStore()
    agent = FakeAgent()

    service = ConversationService(
        store=store,
        agent=agent,
        llm_client=FakeLLM(),
    )

    first_session = str(uuid4())
    second_session = str(uuid4())

    service.ask(
        first_session,
        "How can I open a savings account?",
    )

    service.ask(
        second_session,
        "What is a credit card?",
    )

    assert len(store.get_history(first_session)) == 2
    assert len(store.get_history(second_session)) == 2


def test_invalid_question():
    service = ConversationService(
        store=ConversationStore(),
        agent=FakeAgent(),
        llm_client=FakeLLM(),
    )

    with pytest.raises(ValueError):
        service.ask(str(uuid4()), "   ")
