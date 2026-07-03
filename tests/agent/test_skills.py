from __future__ import annotations

from pathlib import Path

from agent import skills as skills_module
from agent.skills import discover_skills, parse_skill_file


def _write_skill(path: Path, *, name: str, description: str, body: str = "Do the thing.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_parse_skill_file_reads_frontmatter_and_body(tmp_path: Path) -> None:
    skill_file = tmp_path / "my-skill" / "SKILL.md"
    _write_skill(skill_file, name="my-skill", description="Use this when testing.")

    skill = parse_skill_file(skill_file)

    assert skill.name == "my-skill"
    assert skill.description == "Use this when testing."
    assert skill.content == "Do the thing.\n"
    assert skill.file_path == skill_file
    assert skill.disable_model_invocation is False


def test_parse_skill_file_reads_disable_model_invocation(tmp_path: Path) -> None:
    skill_file = tmp_path / "hidden-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: hidden-skill\ndescription: Hidden from the model.\n"
        "disable-model-invocation: true\n---\nBody.\n",
        encoding="utf-8",
    )

    skill = parse_skill_file(skill_file)

    assert skill.disable_model_invocation is True


def test_discover_skills_finds_directory_style_and_loose_md(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    _write_skill(project_skills / "dir-skill" / "SKILL.md", name="dir-skill", description="A.")
    _write_skill(project_skills / "loose-skill.md", name="loose-skill", description="B.")

    found = discover_skills(tmp_path / "project")

    assert {skill.name for skill in found} == {"dir-skill", "loose-skill"}


def test_discover_skills_scans_global_then_project(tmp_path: Path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", global_dir)
    _write_skill(global_dir / "global-skill" / "SKILL.md", name="global-skill", description="G.")
    project_dir = tmp_path / "project"
    _write_skill(
        project_dir / ".haiku" / "skills" / "project-skill" / "SKILL.md",
        name="project-skill",
        description="P.",
    )

    found = discover_skills(project_dir)

    assert {skill.name for skill in found} == {"global-skill", "project-skill"}


def test_discover_skills_deduplicates_by_name_preferring_global(
    tmp_path: Path, monkeypatch
) -> None:
    global_dir = tmp_path / "global"
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", global_dir)
    _write_skill(global_dir / "same" / "SKILL.md", name="same", description="From global.")
    project_dir = tmp_path / "project"
    _write_skill(
        project_dir / ".haiku" / "skills" / "same" / "SKILL.md",
        name="same",
        description="From project.",
    )

    found = discover_skills(project_dir)

    assert len(found) == 1
    assert found[0].description == "From global."


def test_discover_skills_returns_empty_for_missing_directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")

    assert discover_skills(tmp_path / "missing-project") == []


def test_discover_skills_silently_skips_name_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    _write_skill(
        project_skills / "actual-dir-name" / "SKILL.md",
        name="different-name",
        description="Mismatched.",
    )

    found = discover_skills(tmp_path / "project")

    assert found == []


def test_discover_skills_silently_skips_invalid_name_chars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    _write_skill(
        project_skills / "Bad_Name" / "SKILL.md", name="Bad_Name", description="Invalid chars."
    )

    found = discover_skills(tmp_path / "project")

    assert found == []


def test_discover_skills_silently_skips_missing_description(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    skill_file = project_skills / "no-desc" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: no-desc\n---\nBody.\n", encoding="utf-8")

    found = discover_skills(tmp_path / "project")

    assert found == []


def test_discover_skills_silently_skips_malformed_frontmatter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    skill_file = project_skills / "broken" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("no frontmatter here\n", encoding="utf-8")

    found = discover_skills(tmp_path / "project")

    assert found == []
