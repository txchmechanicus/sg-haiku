from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

GLOBAL_SKILLS_DIR = Path.home() / ".haiku" / "skills"
PROJECT_SKILLS_DIR_NAME = Path(".haiku") / "skills"

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_FRONTMATTER_DELIMITER = "---"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    file_path: Path
    disable_model_invocation: bool = False


def discover_skills(cwd: Path) -> list[Skill]:
    skills: list[Skill] = []
    seen_names: set[str] = set()
    for directory in (GLOBAL_SKILLS_DIR, cwd / PROJECT_SKILLS_DIR_NAME):
        for skill in _scan_directory(directory):
            if skill.name in seen_names:
                continue
            seen_names.add(skill.name)
            skills.append(skill)
    return skills


def _scan_directory(directory: Path) -> list[Skill]:
    if not directory.is_dir():
        return []

    found: list[Skill] = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            skill = _try_parse_skill_file(skill_file, expected_name=entry.name)
        elif entry.is_file() and entry.suffix.lower() == ".md":
            skill = _try_parse_skill_file(entry, expected_name=None)
        else:
            continue
        if skill is not None:
            found.append(skill)
    return found


def _try_parse_skill_file(path: Path, *, expected_name: str | None) -> Skill | None:
    try:
        skill = parse_skill_file(path)
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if expected_name is not None and skill.name != expected_name:
        return None
    return skill


def parse_skill_file(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)

    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    _validate_name(name)
    _validate_description(description)

    return Skill(
        name=name,
        description=description,
        content=body,
        file_path=path,
        disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
    )


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    stripped = text.lstrip("﻿")
    if not stripped.startswith(_FRONTMATTER_DELIMITER):
        raise ValueError("Skill file has no frontmatter.")
    remainder = stripped[len(_FRONTMATTER_DELIMITER) :]
    end_index = remainder.find(f"\n{_FRONTMATTER_DELIMITER}")
    if end_index == -1:
        raise ValueError("Skill file frontmatter is not terminated.")
    raw_frontmatter = remainder[:end_index]
    body = remainder[end_index + len(f"\n{_FRONTMATTER_DELIMITER}") :].lstrip("\n")

    parsed = yaml.safe_load(raw_frontmatter)
    if not isinstance(parsed, dict):
        raise ValueError("Skill frontmatter must be a mapping.")
    return parsed, body


def _validate_name(name: str) -> None:
    if not name or len(name) > _MAX_NAME_LENGTH or not _NAME_PATTERN.match(name):
        raise ValueError(f"Invalid skill name: {name!r}")


def _validate_description(description: str) -> None:
    if not description or len(description) > _MAX_DESCRIPTION_LENGTH:
        raise ValueError("Skill description is required and must be <= 1024 characters.")
