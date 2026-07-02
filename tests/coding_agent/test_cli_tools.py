from __future__ import annotations

from pathlib import Path

from coding_agent.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_tools_allowlist_keeps_ls_for_mock_tool_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list files", "--tools", "read, ls"])

    assert result.exit_code == 0
    assert "Tool ls returned" in result.stdout


def test_cli_exclude_tools_removes_ls_from_mock_tool_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list files", "--exclude-tools", "ls"])

    assert result.exit_code == 0
    assert "Mock response: list files" in result.stdout


def test_cli_no_builtin_tools_uses_empty_registry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list files", "--no-builtin-tools"])

    assert result.exit_code == 0
    assert "Mock response: list files" in result.stdout


def test_cli_no_tools_overrides_selection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list files", "--tools", "ls", "--no-tools"])

    assert result.exit_code == 0
    assert "Mock response: list files" in result.stdout


def test_cli_unknown_tool_name_exits_with_config_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--tools", "missing"])

    assert result.exit_code == 2
    assert "Unknown tool name: missing" in result.stderr
