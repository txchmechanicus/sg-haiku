from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter
from upstream.models import Message

from agent.entries import EntryRef
from agent.events import AgentEvent

CURRENT_SESSION_VERSION = 4
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
DEFAULT_SESSION_DIR = Path(".haiku") / "sessions"
_MESSAGE_ADAPTER = TypeAdapter(Message)
_AGENT_EVENT_ADAPTER = TypeAdapter(AgentEvent)

# Entry types that form the session tree (every one carries "id"/"parentId").
# Anything else is treated as forward-compatible/unknown and ignored on load.
_TREE_ENTRY_TYPES = {
    "message",
    "event",
    "model_change",
    "thinking_level_change",
    "compaction",
    "leaf",
    "branch_summary",
}


@dataclass(frozen=True)
class LoadedSession:
    path: Path
    header: dict[str, object]
    entry_refs: list[EntryRef]
    leaf_id: str | None
    entry_ids: frozenset[str] = field(default_factory=frozenset)
    compaction_summary: str | None = None

    @property
    def session_id(self) -> str:
        return str(self.header["id"])

    @property
    def messages(self) -> list[Message]:
        return [ref.message for ref in self.entry_refs]


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
        leaf_id: str | None = None,
        known_ids: frozenset[str] | set[str] | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        validate_session_id(session_id)
        self.session_id = session_id
        self.cwd = cwd.resolve()
        self.path = path
        self._leaf_id = leaf_id
        self._known_ids: set[str] = set(known_ids) if known_ids else set()
        self._id_generator = id_generator
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
        leaf_id: str | None = None,
        known_ids: frozenset[str] | set[str] | None = None,
        id_generator: Callable[[], str] | None = None,
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
            leaf_id=leaf_id,
            known_ids=known_ids,
            id_generator=id_generator,
        )

    def header(self) -> dict[str, object]:
        return dict(self._header)

    def record_message(self, message: Message) -> dict[str, object]:
        return self._append_entry({"type": "message", "message": _dump_message(message)})

    def record_event(self, event: AgentEvent) -> dict[str, object]:
        return self._append_entry(
            {"type": "event", "event": event.model_dump(mode="json", exclude_none=True)}
        )

    def record_model_change(self, *, provider: str, model_id: str) -> dict[str, object]:
        return self._append_entry(
            {"type": "model_change", "provider": provider, "modelId": model_id}
        )

    def record_thinking_level_change(self, *, thinking_level: str) -> dict[str, object]:
        return self._append_entry(
            {"type": "thinking_level_change", "thinkingLevel": thinking_level}
        )

    def record_compaction(
        self,
        *,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
    ) -> dict[str, object]:
        return self._append_entry(
            {
                "type": "compaction",
                "summary": summary,
                "firstKeptEntryId": first_kept_entry_id,
                "tokensBefore": tokens_before,
            }
        )

    def record_leaf_change(
        self, target_id: str, *, summary: str | None = None
    ) -> dict[str, object]:
        """Rewind the active leaf to an earlier entry, leaving the abandoned branch intact."""
        from_id = self._leaf_id
        if summary is not None:
            self._append_entry({"type": "branch_summary", "fromId": from_id, "summary": summary})
        return self._append_entry({"type": "leaf", "targetId": target_id})

    def _append_entry(self, fields: dict[str, object]) -> dict[str, object]:
        entry_id = self._next_entry_id()
        record: dict[str, object] = {
            **fields,
            "id": entry_id,
            "parentId": self._leaf_id,
            "timestamp": _timestamp(),
        }
        self._write(record)
        self._leaf_id = _leaf_id_after_entry(record)
        return record

    def _next_entry_id(self) -> str:
        if self._id_generator is not None:
            entry_id = self._id_generator()
        else:
            entry_id = _generate_random_id(self._known_ids)
        self._known_ids.add(entry_id)
        return entry_id

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
    entries_by_id: dict[str, dict[str, object]] = {}
    current_leaf_id: str | None = None

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

        if entry_type not in _TREE_ENTRY_TYPES:
            continue

        if "id" not in entry:
            raise ValueError(f"Session entry has no id: {path} at line {line_number}.")

        if entry_type == "message":
            if "message" not in entry:
                raise ValueError(f"Session message entry has no message: {path}")
            _MESSAGE_ADAPTER.validate_python(entry["message"])
        elif entry_type == "event":
            if "event" not in entry:
                raise ValueError(f"Session event entry has no event: {path}")
            _AGENT_EVENT_ADAPTER.validate_python(entry["event"])
        elif entry_type == "leaf":
            if "targetId" not in entry:
                raise ValueError(f"Session leaf entry has no targetId: {path}")
        elif entry_type == "compaction":
            if "firstKeptEntryId" not in entry or "summary" not in entry:
                raise ValueError(f"Session compaction entry is missing fields: {path}")

        entries_by_id[str(entry["id"])] = entry
        current_leaf_id = _leaf_id_after_entry(entry)

    if header is None:
        raise ValueError(f"Session file has no session header: {path}")
    if "id" not in header:
        raise ValueError(f"Session header has no id: {path}")
    validate_session_id(str(header["id"]))

    entry_refs, compaction_summary = build_context(entries_by_id, current_leaf_id)

    return LoadedSession(
        path=resolved_path,
        header=header,
        entry_refs=entry_refs,
        leaf_id=current_leaf_id,
        entry_ids=frozenset(entries_by_id),
        compaction_summary=compaction_summary,
    )


def get_path_to_root(
    entries_by_id: dict[str, dict[str, object]], leaf_id: str | None
) -> list[dict[str, object]]:
    if leaf_id is None:
        return []
    path: list[dict[str, object]] = []
    current_id: str | None = leaf_id
    while current_id is not None:
        entry = entries_by_id.get(current_id)
        if entry is None:
            break
        path.append(entry)
        parent_id = entry.get("parentId")
        current_id = str(parent_id) if parent_id is not None else None
    path.reverse()
    return path


def build_context(
    entries_by_id: dict[str, dict[str, object]], leaf_id: str | None
) -> tuple[list[EntryRef], str | None]:
    path = get_path_to_root(entries_by_id, leaf_id)

    last_compaction: dict[str, object] | None = None
    for entry in path:
        if entry["type"] == "compaction":
            last_compaction = entry

    message_entries = [entry for entry in path if entry["type"] == "message"]

    compaction_summary: str | None = None
    cut_index = 0
    if last_compaction is not None:
        compaction_summary = str(last_compaction.get("summary", ""))
        cut_entry_id = str(last_compaction.get("firstKeptEntryId", ""))
        for index, entry in enumerate(message_entries):
            if entry["id"] == cut_entry_id:
                cut_index = index
                break
        else:
            cut_index = 0

    kept = message_entries[cut_index:]
    entry_refs = [
        EntryRef(id=str(entry["id"]), message=_MESSAGE_ADAPTER.validate_python(entry["message"]))
        for entry in kept
    ]
    return entry_refs, compaction_summary


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


def _leaf_id_after_entry(entry: dict[str, object]) -> str | None:
    if entry.get("type") == "leaf":
        target = entry.get("targetId")
        return str(target) if target is not None else None
    entry_id = entry.get("id")
    return str(entry_id) if entry_id is not None else None


def _generate_random_id(known_ids: set[str]) -> str:
    for _ in range(100):
        candidate = uuid4().hex[:8]
        if candidate not in known_ids:
            return candidate
    return uuid4().hex


def _dump_message(message: Message) -> dict[str, object]:
    return message.model_dump(mode="json", exclude_none=True)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
