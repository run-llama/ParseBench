from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .indexer import IndexedDocumentInternal


@dataclass
class SessionState:
    session_id: str
    root_path: Path
    docs_by_id: dict[str, IndexedDocumentInternal]
    created_at: datetime


class AppState:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}

    def create_session(self, root_path: Path, docs_by_id: dict[str, IndexedDocumentInternal]) -> str:
        session_id = uuid4().hex
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            root_path=root_path,
            docs_by_id=docs_by_id,
            created_at=datetime.now(UTC),
        )
        # Keep memory bounded; newest sessions only.
        if len(self.sessions) > 50:
            ordered = sorted(self.sessions.values(), key=lambda s: s.created_at, reverse=True)
            keep = {session.session_id for session in ordered[:50]}
            self.sessions = {sid: state for sid, state in self.sessions.items() if sid in keep}
        return session_id

    def get_session(self, session_id: str) -> SessionState | None:
        return self.sessions.get(session_id)


STATE = AppState()
