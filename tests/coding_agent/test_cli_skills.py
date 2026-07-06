from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from agent import skills as skills_module
from coding_agent.cli import app
from coding_agent.config import ProviderConfig
from typer.testing import CliRunner
from upstream.models import AssistantMessage, AssistantMessageEvent, Message, TextContent
from upstream.providers.base import ModelProvider
from upstream.types import ToolSpec

runner = CliRunner()


class _CapturingProvider(ModelProvider):
    def __init__(self) -> None:
        self.seen_system_prompts: list[str | None] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.seen_system_prompts.append(system_prompt)
        message = AssistantMessage(content=[TextContent(text="ok")], stopReason="stop")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="done", reason="stop", message=message)


def _write_skill(path: Path, *, name: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody.\n", encoding="utf-8"
    )


def test_skills_list_prints_discovered_skills(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    _write_skill(
        tmp_path / ".haiku" / "skills" / "my-skill" / "SKILL.md",
        name="my-skill",
        description="Use this when testing.",
    )

    result = runner.invoke(app, ["skills", "list"])

    assert result.exit_code == 0
    assert "my-skill" in result.stdout
    assert "Use this when testing." in result.stdout


def test_skills_list_reports_none_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")

    result = runner.invoke(app, ["skills", "list"])

    assert result.exit_code == 0
    assert "No skills found." in result.stdout


def test_skills_list_prints_diagnostics_for_loaded_but_warned_skill(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    _write_skill(
        tmp_path / ".haiku" / "skills" / "Bad_Name" / "SKILL.md",
        name="Bad_Name",
        description="Invalid chars.",
    )

    result = runner.invoke(app, ["skills", "list"])

    assert result.exit_code == 0
    assert "Bad_Name" in result.stdout
    assert "[Skill conflicts]" in result.stdout
    assert "invalid characters" in result.stdout


def test_cli_includes_available_skills_in_system_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    _write_skill(
        tmp_path / ".haiku" / "skills" / "my-skill" / "SKILL.md",
        name="my-skill",
        description="Use this when testing.",
    )
    provider = _CapturingProvider()

    async def _build(self):
        return provider

    monkeypatch.setattr(ProviderConfig, "build", _build)

    result = runner.invoke(app, ["hello"])

    assert result.exit_code == 0
    assert "<available_skills>" in provider.seen_system_prompts[0]
    assert "my-skill" in provider.seen_system_prompts[0]


def test_cli_no_skills_flag_omits_skills_block(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    _write_skill(
        tmp_path / ".haiku" / "skills" / "my-skill" / "SKILL.md",
        name="my-skill",
        description="Use this when testing.",
    )
    provider = _CapturingProvider()

    async def _build(self):
        return provider

    monkeypatch.setattr(ProviderConfig, "build", _build)

    result = runner.invoke(app, ["hello", "--no-skills"])

    assert result.exit_code == 0
    assert "<available_skills>" not in (provider.seen_system_prompts[0] or "")
