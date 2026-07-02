from __future__ import annotations

import json
from pathlib import Path

from coding_agent.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_mock_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello"])

    assert result.exit_code == 0
    assert "Mock response: hello" in result.stdout


def test_cli_mock_tool_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list files"])

    assert result.exit_code == 0
    assert "Tool ls returned" in result.stdout


def test_cli_openai_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["hello", "--provider", "openai-compatible", "--model", "gpt-4o-mini"],
        env={"OPENAI_API_KEY": ""},
    )

    assert result.exit_code == 2
    assert "requires --api-key or OPENAI_API_KEY" in result.stderr


def test_cli_list_models_includes_builtins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--list-models"])

    assert result.exit_code == 0
    assert "provider\tid\tname\tapi\tcontextWindow\tmaxTokens" in result.stdout
    assert "openai-compatible\tgpt-4o-mini" in result.stdout


def test_cli_list_models_does_not_include_mock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--list-models"])

    assert result.exit_code == 0
    assert "mock" not in result.stdout


def test_cli_list_models_reads_explicit_json(tmp_path: Path, monkeypatch) -> None:
    import json as _json

    monkeypatch.chdir(tmp_path)
    Path("models.json").write_text(
        _json.dumps({
            "providers": {
                "local": {
                    "api": "openai-completions",
                    "baseUrl": "http://localhost:11434/v1",
                    "models": [
                        {"id": "llama", "name": "Llama", "contextWindow": 4096, "maxTokens": 512}
                    ],
                }
            }
        }),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["--models-config", "models.json", "--list-models"])

    assert result.exit_code == 0
    assert "local\tllama\tLlama\topenai-completions\t4096\t512" in result.stdout


def test_cli_provider_without_model_exits_with_config_error(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--provider", "openai-compatible"])

    assert result.exit_code == 2
    assert "requires --model" in result.stderr


def test_cli_mock_provider_is_not_selectable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result_model = runner.invoke(app, ["hello", "--model", "mock"])
    result_provider = runner.invoke(app, ["hello", "--provider", "mock", "--model", "anything"])

    assert result_model.exit_code == 2
    assert "dev-only" in result_model.stderr
    assert result_provider.exit_code == 2
    assert "dev-only" in result_provider.stderr


def test_cli_auth_set_list_status_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    auth_file = tmp_path / "auth.json"

    set_result = runner.invoke(
        app,
        [
            "auth",
            "set",
            "openai-compatible",
            "--api-key",
            "sk-secret-value",
            "--auth-file",
            str(auth_file),
        ],
    )
    list_result = runner.invoke(app, ["auth", "list", "--auth-file", str(auth_file)])
    status_result = runner.invoke(
        app,
        ["auth", "status", "openai-compatible", "--auth-file", str(auth_file)],
    )
    unset_result = runner.invoke(
        app,
        ["auth", "unset", "openai-compatible", "--auth-file", str(auth_file)],
    )
    empty_result = runner.invoke(app, ["auth", "list", "--auth-file", str(auth_file)])

    assert set_result.exit_code == 0
    assert "sk-secret-value" not in set_result.stdout
    assert list_result.exit_code == 0
    assert list_result.stdout.split() == ["openai-compatible", "api_key", "auth_file"]
    assert "sk-secret-value" not in list_result.stdout
    assert status_result.exit_code == 0
    assert "sk-s...alue" in status_result.stdout
    assert "sk-secret-value" not in status_result.stdout
    assert unset_result.exit_code == 0
    assert empty_result.exit_code == 0
    assert "No auth entries." in empty_result.stdout


def test_cli_json_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--mode", "json"])

    assert result.exit_code == 0
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert records[0]["type"] == "session"
    assert records[0]["version"] == 4
    assert records[1]["type"] == "agent_start"
    assert records[-1]["type"] == "agent_end"


def test_cli_rpc_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--mode", "rpc"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"] == "Mock response: hello"
    assert payload["errors"] == []


def test_cli_json_mode_includes_tool_events(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list files", "--mode", "json"])

    assert result.exit_code == 0
    records = [json.loads(line) for line in result.stdout.splitlines()]
    types = [record["type"] for record in records]
    assert types[0] == "session"
    assert "tool_execution_start" in types
    assert "tool_execution_end" in types
    assert types[-1] == "agent_end"


def test_cli_applies_prompt_template(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("template.txt").write_text("Wrapped: {prompt}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["hello", "--prompt-template", "template.txt", "--no-session"],
    )

    assert result.exit_code == 0
    assert "Mock response: Wrapped: hello" in result.stdout


def test_cli_no_prompt_templates_ignores_prompt_template(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("template.txt").write_text("Wrapped: {prompt}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "hello",
            "--prompt-template",
            "template.txt",
            "--no-prompt-templates",
            "--no-session",
        ],
    )

    assert result.exit_code == 0
    assert "Mock response: hello" in result.stdout


def test_cli_missing_prompt_template_exits_with_config_error(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--prompt-template", "missing.txt"])

    assert result.exit_code == 2
    assert "Prompt template does not exist" in result.stderr
