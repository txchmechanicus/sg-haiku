from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter
from upstream.models import Message

from agent.events import AgentEvent

CURRENT_SESSION_VERSION = 3
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
DEFAULT_SESSION_DIR = Path(".haiku") / "sessions"
_MESSAGE_ADAPTER = TypeAdapter(Message)
_AGENT_EVENT_ADAPTER = TypeAdapter(AgentEvent)


@dataclass(frozen=True)
class LoadedSession:
    path: Path
    header: dict[str, object]
    messages: list[Message]

    @property
    def session_id(self) -> str:
        return str(self.header["id"])


class SessionManager:
    def __init__(
        self,
        *,
        session_id: str,
        cwd: Path,
        path: Path | None,
        parent_session: str | None = None,
        session_name: str | None = None,
        append: bool = False,
        header: dict[str, object] | None = None,
    ) -> None:
        validate_session_id(session_id)
        self.session_id = session_id
        self.cwd = cwd.resolve()
        self.path = path
        self._header = header or {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": self.session_id,
            "timestamp": _timestamp(),
            "cwd": str(self.cwd),
        }
        if header is None and parent_session is not None:
            self._header["parentSession"] = parent_session
        if header is None and session_name is not None:
            self._header["name"] = session_name
        if self.path is not None and not append:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write(self.header())

    @classmethod
    def create(
        cls,
        *,
        explicit_path: Path | None = None,
        session_dir: Path | None = None,
        session_id: str | None = None,
        session_name: str | None = None,
        cwd: Path | None = None,
        write_enabled: bool = True,
        parent_session: str | None = None,
        append: bool = False,
        header: dict[str, object] | None = None,
    ) -> SessionManager:
        resolved_id = session_id or str(uuid4())
        path = explicit_path
        if path is None and write_enabled:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            path = (session_dir or DEFAULT_SESSION_DIR) / f"{stamp}-{resolved_id[:8]}.jsonl"
        if not write_enabled:
            path = None
        return cls(
            session_id=resolved_id,
            session_name=session_name,
            cwd=cwd or Path.cwd(),
            path=path,
            parent_session=parent_session,
            append=append,
            header=header,
        )

    def header(self) -> dict[str, object]:
        return dict(self._header)

    def record_message(self, message: Message) -> dict[str, object]:
        record = {"type": "message", "message": _dump_message(message)}
        self._write(record)
        return record

    def record_event(self, event: AgentEvent) -> dict[str, object]:
        record = {"type": "event", "event": event.model_dump(mode="json", exclude_none=True)}
        self._write(record)
        return record

    def record_model_change(self, *, provider: str, model_id: str) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "model_change",
            "provider": provider,
            "modelId": model_id,
        }
        self._write(record)
        return record

    def record_thinking_level_change(self, *, thinking_level: str) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "thinking_level_change",
            "thinkingLevel": thinking_level,
        }
        self._write(record)
        return record

    def record_compaction(
        self,
        *,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "compaction",
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }
        self._write(record)
        return record

    def _write(self, payload: dict[str, object]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def validate_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError(
            "Session id must start and end with an alphanumeric character and contain "
            "only alphanumeric characters, '.', '_', or '-'."
        )


def load_session(path: Path) -> LoadedSession:
    resolved_path = path.expanduser()
    if not resolved_path.exists():
        raise ValueError(f"Session file does not exist: {path}")
    if not resolved_path.is_file():
        raise ValueError(f"Session path is not a file: {path}")

    header: dict[str, object] | None = None
    messages: list[Message] = []
    for line_number, line in enumerate(resolved_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in session {path} at line {line_number}.") from exc

        if not isinstance(entry, dict):
            raise ValueError(f"Invalid session entry in {path} at line {line_number}.")
        entry_type = entry.get("type")
        if entry_type == "session":
            if header is None:
                if "id" not in entry:
                    raise ValueError(f"Session header has no id: {path}")
                header = entry
            continue
        if entry_type == "message":
            if "message" not in entry:
                raise ValueError(f"Session message entry has no message: {path}")
            messages.append(_MESSAGE_ADAPTER.validate_python(entry["message"]))
            continue
        if entry_type == "event":
            if "event" not in entry:
                raise ValueError(f"Session event entry has no event: {path}")
            _AGENT_EVENT_ADAPTER.validate_python(entry["event"])
            continue

    if header is None:
        raise ValueError(f"Session file has no session header: {path}")
    if "id" not in header:
        raise ValueError(f"Session header has no id: {path}")
    validate_session_id(str(header["id"]))
    return LoadedSession(path=resolved_path, header=header, messages=messages)


def find_sessions(session_dir: Path) -> list[LoadedSession]:
    directory = session_dir.expanduser()
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"Session directory is not a directory: {session_dir}")

    sessions: list[LoadedSession] = []
    for path in directory.glob("*.jsonl"):
        sessions.append(load_session(path))
    return sorted(
        sessions,
        key=lambda session: session.path.stat().st_mtime,
        reverse=True,
    )


def latest_session(session_dir: Path) -> LoadedSession:
    sessions = find_sessions(session_dir)
    if not sessions:
        raise ValueError(f"No sessions found in {session_dir}")
    return sessions[0]


def resolve_session_reference(reference: str, session_dir: Path) -> LoadedSession:
    path = Path(reference).expanduser()
    if path.exists():
        return load_session(path)

    sessions = find_sessions(session_dir)
    exact = [session for session in sessions if session.session_id == reference]
    if len(exact) == 1:
        return exact[0]

    partial = [session for session in sessions if session.session_id.startswith(reference)]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        ids = ", ".join(sorted(session.session_id for session in partial))
        raise ValueError(f"Ambiguous session id '{reference}'. Matches: {ids}")
    raise ValueError(f"Session not found: {reference}")


def _dump_message(message: Message) -> dict[str, object]:
    return message.model_dump(mode="json", exclude_none=True)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
