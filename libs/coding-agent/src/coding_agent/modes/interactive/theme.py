from __future__ import annotations

import json
from pathlib import Path

from textual.theme import Theme

# Ported from ~/code/stgzr/apps/aip's shared design system
# (starship/src/styles/starship.css `.dark { ... }` block, "inspired by Claude.ai").
# Only the values that theme actually defines are set explicitly here; anything it
# leaves unspecified (success/warning) uses a close Tailwind-palette match instead of
# Textual's defaults, to stay in the same visual family.
STARGAZER_DARK = Theme(
    name="stargazer-dark",
    dark=True,
    background="#1a1a1a",
    surface="#242424",
    panel="#2A2A2A",
    boost="#2E2E2E",
    primary="#c2a4f0",
    secondary="#7B5FC7",
    accent="#c2a4f0",
    foreground="#fafafa",
    error="#F87171",
    success="#4ADE80",
    warning="#FBBF24",
    variables={
        "border": "#3A3A3A",
        "text-muted": "#a3a3a3",
        "input-selection-background": "#643fb2 35%",
    },
)

# Same `~/.haiku/...` (global) + `.haiku/...` (project) convention as
# `coding_agent.extensions.loader`/`agent.skills` — one JSON file per theme.
GLOBAL_THEMES_DIR = Path.home() / ".haiku" / "themes"
PROJECT_THEMES_DIR_NAME = Path(".haiku") / "themes"

# The subset of Pi's much larger theme schema that Textual's `Theme` model (and our
# current CSS) actually consumes — not the full 51-token format (we don't render
# markdown, syntax highlighting, thinking levels, or a bash-mode indicator yet).
_THEME_FIELDS = frozenset(
    {
        "name",
        "dark",
        "background",
        "surface",
        "panel",
        "boost",
        "primary",
        "secondary",
        "accent",
        "foreground",
        "error",
        "success",
        "warning",
        "variables",
    }
)


def discover_theme_files(cwd: Path) -> list[Path]:
    """Project-local theme files first, then global — matches `extensions.loader`'s
    precedence (first occurrence wins on a name collision, applied by
    `discover_custom_themes`), so a project can override a same-named global theme."""
    resolved_cwd = cwd.resolve()
    files: list[Path] = []
    for directory in (resolved_cwd / PROJECT_THEMES_DIR_NAME, GLOBAL_THEMES_DIR):
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.json")))
    return files


def load_theme_file(path: Path) -> Theme | None:
    """Parses one theme JSON file into a `Theme`. Returns `None` on any problem (missing/
    invalid JSON, no `name`, unknown-shaped values) rather than raising — a single bad file
    must never abort discovery of the others, matching `extensions`/`skills` isolation."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("name"):
        return None
    kwargs = {key: value for key, value in data.items() if key in _THEME_FIELDS}
    try:
        return Theme(**kwargs)
    except (TypeError, ValueError):
        return None


def discover_custom_themes(cwd: Path) -> tuple[list[Theme], list[str]]:
    """Loads every discovered theme file. Returns `(themes, warnings)` — `warnings` are
    human-readable strings for files that failed to load, meant to be printed once before
    the TUI starts (same pattern as extension load errors)."""
    themes: list[Theme] = []
    warnings: list[str] = []
    seen_names: set[str] = set()
    for path in discover_theme_files(cwd):
        theme = load_theme_file(path)
        if theme is None:
            warnings.append(f"{path}: invalid theme file, skipped")
            continue
        if theme.name in seen_names:
            continue  # first occurrence wins: project-local already added, keep it
        seen_names.add(theme.name)
        themes.append(theme)
    return themes, warnings
