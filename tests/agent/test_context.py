from __future__ import annotations

from pathlib import Path

import pytest
from agent import skills as skills_module
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


def _write_skill(path: Path, *, name: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody.\n", encoding="utf-8"
    )


def test_context_builder_includes_available_skills_block(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    skill_file = tmp_path / ".haiku" / "skills" / "my-skill" / "SKILL.md"
    _write_skill(skill_file, name="my-skill", description="Use this when testing.")
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello")

    assert len(context.skills) == 1
    assert "<available_skills>" in context.system_prompt
    assert "<name>my-skill</name>" in context.system_prompt
    assert "<description>Use this when testing.</description>" in context.system_prompt
    assert str(skill_file) in context.system_prompt


def test_context_builder_can_disable_skills(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    _write_skill(
        tmp_path / ".haiku" / "skills" / "my-skill" / "SKILL.md",
        name="my-skill",
        description="Use this when testing.",
    )
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello", include_skills=False)

    assert context.skills == []
    assert "<available_skills>" not in context.system_prompt


def test_context_builder_omits_skills_block_when_none_discovered(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello")

    assert context.system_prompt == "base"


def test_context_builder_excludes_disabled_skills_from_model_visible_block(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    skill_file = tmp_path / ".haiku" / "skills" / "hidden-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: hidden-skill\ndescription: Hidden.\n"
        "disable-model-invocation: true\n---\nBody.\n",
        encoding="utf-8",
    )
    builder = PromptContextBuilder(cwd=tmp_path, base_system_prompt="base")

    context = builder.build(prompt="hello")

    assert len(context.skills) == 1
    assert "<available_skills>" not in context.system_prompt
