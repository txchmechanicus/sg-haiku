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

    skill, diagnostics = parse_skill_file(skill_file)

    assert skill is not None
    assert skill.name == "my-skill"
    assert skill.description == "Use this when testing."
    assert skill.content == "Do the thing.\n"
    assert skill.file_path == skill_file
    assert skill.disable_model_invocation is False
    assert diagnostics == []


def test_parse_skill_file_reads_disable_model_invocation(tmp_path: Path) -> None:
    skill_file = tmp_path / "hidden-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: hidden-skill\ndescription: Hidden from the model.\n"
        "disable-model-invocation: true\n---\nBody.\n",
        encoding="utf-8",
    )

    skill, _diagnostics = parse_skill_file(skill_file)

    assert skill is not None
    assert skill.disable_model_invocation is True


def test_discover_skills_finds_directory_style_and_loose_md(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    _write_skill(project_skills / "dir-skill" / "SKILL.md", name="dir-skill", description="A.")
    _write_skill(project_skills / "loose-skill.md", name="loose-skill", description="B.")

    found, diagnostics = discover_skills(tmp_path / "project")

    assert {skill.name for skill in found} == {"dir-skill", "loose-skill"}
    assert diagnostics == []


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

    found, diagnostics = discover_skills(project_dir)

    assert {skill.name for skill in found} == {"global-skill", "project-skill"}
    assert diagnostics == []


def test_discover_skills_reports_collision_preferring_global(tmp_path: Path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", global_dir)
    global_skill = global_dir / "same" / "SKILL.md"
    _write_skill(global_skill, name="same", description="From global.")
    project_dir = tmp_path / "project"
    project_skill = project_dir / ".haiku" / "skills" / "same" / "SKILL.md"
    _write_skill(project_skill, name="same", description="From project.")

    found, diagnostics = discover_skills(project_dir)

    assert len(found) == 1
    assert found[0].description == "From global."
    assert len(diagnostics) == 1
    assert diagnostics[0].type == "collision"
    assert str(global_skill) in diagnostics[0].message
    assert str(project_skill) in diagnostics[0].message
    assert diagnostics[0].path == project_skill


def test_discover_skills_scans_extra_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_dir = tmp_path / "project"
    extra_dir = tmp_path / "extension-skills"
    _write_skill(extra_dir / "extra-skill" / "SKILL.md", name="extra-skill", description="E.")
    loose_file = tmp_path / "loose.md"
    _write_skill(loose_file, name="loose-skill", description="L.")

    found, diagnostics = discover_skills(project_dir, [extra_dir, loose_file])

    assert {skill.name for skill in found} == {"extra-skill", "loose-skill"}
    assert diagnostics == []


def test_discover_skills_extra_path_never_overrides_project_or_global(
    tmp_path: Path, monkeypatch
) -> None:
    global_dir = tmp_path / "global"
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", global_dir)
    _write_skill(global_dir / "same" / "SKILL.md", name="same", description="From global.")
    project_dir = tmp_path / "project"
    extra_dir = tmp_path / "extension-skills"
    _write_skill(extra_dir / "same" / "SKILL.md", name="same", description="From extension.")

    found, diagnostics = discover_skills(project_dir, [extra_dir])

    assert len(found) == 1
    assert found[0].description == "From global."
    assert len(diagnostics) == 1
    assert diagnostics[0].type == "collision"


def test_discover_skills_warns_on_missing_extra_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_dir = tmp_path / "project"

    found, diagnostics = discover_skills(project_dir, [tmp_path / "nope"])

    assert found == []
    assert len(diagnostics) == 1
    assert diagnostics[0].type == "warning"
    assert "does not exist" in diagnostics[0].message


def test_discover_skills_returns_empty_for_missing_directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")

    assert discover_skills(tmp_path / "missing-project") == ([], [])


def test_discover_skills_loads_skill_with_invalid_name_but_warns(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    _write_skill(
        project_skills / "Bad_Name" / "SKILL.md", name="Bad_Name", description="Invalid chars."
    )

    found, diagnostics = discover_skills(tmp_path / "project")

    assert len(found) == 1
    assert found[0].name == "Bad_Name"
    assert len(diagnostics) == 1
    assert diagnostics[0].type == "warning"
    assert "invalid characters" in diagnostics[0].message


def test_discover_skills_loads_skill_with_hyphen_issues_but_warns(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    _write_skill(
        project_skills / "leading" / "SKILL.md", name="-leading", description="Leading hyphen."
    )

    found, diagnostics = discover_skills(tmp_path / "project")

    assert len(found) == 1
    assert any("start or end with a hyphen" in d.message for d in diagnostics)


def test_discover_skills_drops_skill_with_missing_description(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    skill_file = project_skills / "no-desc" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: no-desc\n---\nBody.\n", encoding="utf-8")

    found, diagnostics = discover_skills(tmp_path / "project")

    assert found == []
    assert len(diagnostics) == 1
    assert diagnostics[0].type == "warning"
    assert diagnostics[0].message == "description is required"


def test_discover_skills_loads_skill_with_long_description_but_warns(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    long_description = "x" * 1025
    _write_skill(
        project_skills / "long-desc" / "SKILL.md", name="long-desc", description=long_description
    )

    found, diagnostics = discover_skills(tmp_path / "project")

    assert len(found) == 1
    assert found[0].description == long_description
    assert any("exceeds 1024 characters" in d.message for d in diagnostics)


def test_discover_skills_drops_skill_with_malformed_frontmatter(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(skills_module, "GLOBAL_SKILLS_DIR", tmp_path / "missing-global")
    project_skills = tmp_path / "project" / ".haiku" / "skills"
    skill_file = project_skills / "broken" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("no frontmatter here\n", encoding="utf-8")

    found, diagnostics = discover_skills(tmp_path / "project")

    assert found == []
    assert len(diagnostics) == 1
    assert diagnostics[0].type == "warning"
