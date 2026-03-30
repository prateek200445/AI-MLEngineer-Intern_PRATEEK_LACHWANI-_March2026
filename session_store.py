from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any
import uuid


@dataclass
class PendingClarification:
    original_question: str
    target_course: str | None
    missing_fields: list[str] = field(default_factory=list)
    follow_up_questions: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionState:
    session_id: str
    current_user_context: dict[str, Any] = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class InMemorySessionStore:
    def __init__(self, ttl_minutes: int = 45):
        self.ttl = timedelta(minutes=max(1, int(ttl_minutes)))
        self._store: dict[str, SessionState] = {}
        self._lock = Lock()

    def _is_expired(self, session: SessionState) -> bool:
        return datetime.now(timezone.utc) - session.updated_at > self.ttl

    def cleanup_expired(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [
                session_id
                for session_id, session in self._store.items()
                if now - session.updated_at > self.ttl
            ]
            for session_id in expired:
                self._store.pop(session_id, None)

    def get_or_create(self, session_id: str | None) -> SessionState:
        self.cleanup_expired()

        with self._lock:
            if session_id and session_id in self._store:
                session = self._store[session_id]
                if not self._is_expired(session):
                    session.touch()
                    return session
                self._store.pop(session_id, None)

            sid = session_id or str(uuid.uuid4())
            session = SessionState(session_id=sid)
            self._store[sid] = session
            return session


SESSION_STORE = InMemorySessionStore(ttl_minutes=45)
