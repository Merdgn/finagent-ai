
from uuid import uuid4

import pytest

from app.memory.conversation_store import ConversationStore


def test_message_history():
    store = ConversationStore()
    session = str(uuid4())

    store.add_message(
        session,
        "user",
        "How can I open an account?",
    )

    store.add_message(
        session,
        "assistant",
        "Identity verification is required.",
    )

    history = store.get_history(session)

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_session_isolation():
    store = ConversationStore()

    first = str(uuid4())
    second = str(uuid4())

    store.add_message(first, "user", "First user")
    store.add_message(second, "user", "Second user")

    first_history = store.get_history(first)
    second_history = store.get_history(second)

    assert first_history[0].content == "First user"
    assert second_history[0].content == "Second user"


def test_memory_limit():
    store = ConversationStore(max_messages=2)
    session = str(uuid4())

    store.add_message(session, "user", "First")
    store.add_message(session, "assistant", "Second")
    store.add_message(session, "user", "Third")

    history = store.get_history(session)

    assert len(history) == 2
    assert history[0].content == "Second"
    assert history[1].content == "Third"


def test_clear_session():
    store = ConversationStore()
    session = str(uuid4())

    store.add_message(session, "user", "Hello")
    store.clear_session(session)

    assert store.get_history(session) == []


def test_invalid_session():
    store = ConversationStore()

    with pytest.raises(ValueError):
        store.add_message(
            "invalid-id",
            "user",
            "Hello",
        )


def test_session_capacity():
    store = ConversationStore(max_sessions=1)

    store.add_message(str(uuid4()), "user", "Hello")

    with pytest.raises(RuntimeError):
        store.add_message(str(uuid4()), "user", "Hello")
