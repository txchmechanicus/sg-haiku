from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

GLOBAL_SKILLS_DIR = Path.home() / ".haiku" / "skills"
PROJECT_SKILLS_DIR_NAME = Path(".haiku") / "skills"

_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_FRONTMATTER_DELIMITER = "---"
_NAME_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    file_path: Path
    disable_model_invocation: bool = False


@dataclass(frozen=True)
class SkillDiagnostic:
    type: Literal["warning", "error", "collision"]
    message: str
    path: Path


def discover_skills(cwd: Path) -> tuple[list[Skill], list[SkillDiagnostic]]:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    first_seen_path: dict[str, Path] = {}

    for directory in (GLOBAL_SKILLS_DIR, cwd / PROJECT_SKILLS_DIR_NAME):
        dir_skills, dir_diagnostics = _scan_directory(directory)
        diagnostics.extend(dir_diagnostics)
        for skill in dir_skills:
            if skill.name in first_seen_path:
                diagnostics.append(
                    SkillDiagnostic(
                        type="collision",
                        message=(
                            f'skill "{skill.name}" already defined at '
                            f'{first_seen_path[skill.name]}; ignoring duplicate at '
                            f'{skill.file_path}'
                        ),
                        path=skill.file_path,
                    )
                )
                continue
            first_seen_path[skill.name] = skill.file_path
            skills.append(skill)

    return skills, diagnostics


def _scan_directory(directory: Path) -> tuple[list[Skill], list[SkillDiagnostic]]:
    if not directory.is_dir():
        return [], []

    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            skill, file_diagnostics = parse_skill_file(skill_file)
        elif entry.is_file() and entry.suffix.lower() == ".md":
            skill, file_diagnostics = parse_skill_file(entry)
        else:
            continue
        diagnostics.extend(file_diagnostics)
        if skill is not None:
            skills.append(skill)
    return skills, diagnostics


def parse_skill_file(path: Path) -> tuple[Skill | None, list[SkillDiagnostic]]:
    try:
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        message = str(exc) or "failed to parse skill file"
        return None, [SkillDiagnostic(type="warning", message=message, path=path)]

    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()

    messages: list[str] = []
    if not name:
        return None, [SkillDiagnostic(type="warning", message="name is required", path=path)]
    messages.extend(_name_warnings(name))

    description_messages, description_fatal = _description_result(description)
    messages.extend(description_messages)
    diagnostics = [SkillDiagnostic(type="warning", message=m, path=path) for m in messages]

    if description_fatal:
        return None, diagnostics

    skill = Skill(
        name=name,
        description=description,
        content=body,
        file_path=path,
        disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
    )
    return skill, diagnostics


def _name_warnings(name: str) -> list[str]:
    warnings: list[str] = []
    if len(name) > _MAX_NAME_LENGTH:
        warnings.append(f"name exceeds {_MAX_NAME_LENGTH} characters ({len(name)})")
    if not all(char in _NAME_ALLOWED_CHARS for char in name):
        warnings.append(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
        )
    if name.startswith("-") or name.endswith("-"):
        warnings.append("name must not start or end with a hyphen")
    if "--" in name:
        warnings.append("name must not contain consecutive hyphens")
    return warnings


def _description_result(description: str) -> tuple[list[str], bool]:
    if not description:
        return ["description is required"], True
    if len(description) > _MAX_DESCRIPTION_LENGTH:
        return [
            f"description exceeds {_MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
        ], False
    return [], False


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
