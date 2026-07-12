from __future__ import annotations

import json
from pathlib import Path

from coding_agent.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def _session_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_interactive_creates_session_file(tmp_path: Path, monkeypatch) -> None:
    """Bare `haiku` (no prompt) now threads --session/--session-dir into interactive mode
    instead of silently ignoring them. `CliRunner` has no real tty, so the TUI itself never
    starts (`ProcessTerminal.enter_raw_mode()` rejects non-tty stdin) — but the session file
    is created and its header written before that check runs, which is enough to prove the
    flags are no longer dropped on this code path."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--session", "session.jsonl"])

    assert result.exit_code == 2
    records = _session_records(Path("session.jsonl"))
    assert records[0]["type"] == "session"
    assert records[0]["version"] == 4


def test_cli_interactive_no_session_does_not_write_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["--no-session"])

    assert not Path(".haiku").exists()


def test_cli_interactive_rejects_multiple_session_load_modes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--continue", "--resume", "some-id"])

    assert result.exit_code == 2
    assert "Use only one of --continue, --resume, or --fork" in result.output


def test_cli_interactive_continue_appends_latest_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session_dir = Path(".haiku") / "sessions"
    session_dir.mkdir(parents=True)
    existing = session_dir / "20200101T000000Z-abcd1234.jsonl"
    existing.write_text(
        json.dumps({"type": "session", "version": 4, "id": "abcd1234", "timestamp": "t"}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["--continue"])

    assert result.exit_code == 2  # non-tty TUI start still fails, same as above
    records = _session_records(existing)
    assert records[0]["id"] == "abcd1234"  # header untouched (append mode)
