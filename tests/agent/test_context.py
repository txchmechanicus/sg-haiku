from __future__ import annotations

from pathlib import Path

import pytest
from agent.context import PromptContextBuilder


def test_context_builder_discovers_context_files_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("claude context", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("agent context", encoding="utf-8")
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello")

    assert [path.name for path in context.context_files] == ["AGENTS.md", "CLAUDE.md"]
    assert context.system_prompt == (
        "base\n\n"
        "Context from AGENTS.md:\nagent context\n\n"
        "Context from CLAUDE.md:\nclaude context"
    )


def test_context_builder_can_disable_context_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("agent context", encoding="utf-8")
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello", include_context_files=False)

    assert context.context_files == []
    assert context.system_prompt == "base"


def test_context_builder_overrides_system_prompt_from_literal(tmp_path: Path) -> None:
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello", system_prompt="custom")

    assert context.system_prompt == "custom"


def test_context_builder_reads_system_prompt_from_file(tmp_path: Path) -> None:
    (tmp_path / "SYSTEM.md").write_text("custom from file", encoding="utf-8")
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello", system_prompt="SYSTEM.md")

    assert context.system_prompt == "custom from file"


def test_context_builder_appends_system_prompts(tmp_path: Path) -> None:
    (tmp_path / "append.md").write_text("file append", encoding="utf-8")
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(
        prompt="hello",
        append_system_prompts=["literal append", "append.md"],
    )

    assert context.system_prompt == "base\n\nliteral append\n\nfile append"


def test_context_builder_applies_prompt_template_placeholder(tmp_path: Path) -> None:
    (tmp_path / "template.txt").write_text("Task: {prompt}\n", encoding="utf-8")
    builder = PromptContextBuilder(cwd=tmp_path)

    context = builder.build(prompt="hello", prompt_template=tmp_path / "template.txt")

    assert context.prompt == "Task: hello\n"


def test_context_builder_applies_prompt_template_without_placeholder(tmp_path: Path) -> None:
    (tmp_path / "template.txt").write_text("Instructions\n", encoding="utf-8")
    builder = PromptContextBuilder(cwd=tmp_path)

    context = builder.build(prompt="hello", prompt_template=tmp_path / "template.txt")

    assert context.prompt == "Instructions\n\nhello"


def test_context_builder_can_disable_prompt_template(tmp_path: Path) -> None:
    (tmp_path / "template.txt").write_text("Task: {prompt}", encoding="utf-8")
    builder = PromptContextBuilder(cwd=tmp_path)

    context = builder.build(
        prompt="hello",
        prompt_template=tmp_path / "template.txt",
        use_prompt_templates=False,
    )

    assert context.prompt == "hello"


def test_context_builder_rejects_missing_prompt_template(tmp_path: Path) -> None:
    builder = PromptContextBuilder(cwd=tmp_path)

    with pytest.raises(ValueError, match="Prompt template does not exist"):
        builder.build(prompt="hello", prompt_template=tmp_path / "missing.txt")
