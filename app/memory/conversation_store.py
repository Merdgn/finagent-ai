
from collections import deque
from dataclasses import dataclass
from threading import Lock
from uuid import UUID


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ConversationStore:
    """
    Temporary in-memory conversation storage.

    For local development only.
    Not a persistent or production-ready database.
    """

    def __init__(
        self,
        max_messages: int = 10,
        max_sessions: int = 100,
    ):
        if max_messages < 2 or max_sessions < 1:
            raise ValueError("Invalid memory limits")

        self.max_messages = max_messages
        self.max_sessions = max_sessions

        self._sessions: dict[str, deque] = {}
        self._lock = Lock()

    def _validate_session(self, session_id: str):
        try:
            return str(UUID(session_id))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Invalid session ID")

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        session_id = self._validate_session(session_id)

        if role not in {"user", "assistant"}:
            raise ValueError("Invalid message role")

        if not content or not content.strip():
            raise ValueError("Message cannot be empty")

        if len(content) > 4000:
            raise ValueError("Message too long")

        message = ChatMessage(
            role=role,
            content=content.strip(),
        )

        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self.max_sessions:
                    raise RuntimeError("Session capacity reached")

                self._sessions[session_id] = deque(
                    maxlen=self.max_messages
                )

            self._sessions[session_id].append(message)

    def get_history(
        self,
        session_id: str,
    ) -> list[ChatMessage]:
        session_id = self._validate_session(session_id)

        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        session_id = self._validate_session(session_id)

        with self._lock:
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
