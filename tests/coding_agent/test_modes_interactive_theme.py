from __future__ import annotations

import json
from pathlib import Path

from coding_agent.modes.interactive.theme import (
    discover_custom_themes,
    discover_theme_files,
    load_theme_file,
)


def _write_theme(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")


def test_load_theme_file_maps_known_fields(tmp_path: Path) -> None:
    path = tmp_path / "custom.json"
    _write_theme(
        path,
        name="custom",
        dark=True,
        primary="#ff0000",
        variables={"border": "#00ff00"},
        unknown_field="ignored",
    )

    theme = load_theme_file(path)

    assert theme is not None
    assert theme.name == "custom"
    assert theme.primary == "#ff0000"


def test_load_theme_file_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")

    assert load_theme_file(path) is None


def test_load_theme_file_rejects_missing_name(tmp_path: Path) -> None:
    path = tmp_path / "noname.json"
    _write_theme(path, primary="#ff0000")

    assert load_theme_file(path) is None


def test_discover_theme_files_finds_project_and_global(tmp_path: Path, monkeypatch) -> None:
    import coding_agent.modes.interactive.theme as theme_module

    home = tmp_path / "home"
    project = tmp_path / "project"
    monkeypatch.setattr(theme_module, "GLOBAL_THEMES_DIR", home / ".haiku" / "themes")
    _write_theme(home / ".haiku" / "themes" / "global-theme.json", name="global-theme")
    _write_theme(project / ".haiku" / "themes" / "project-theme.json", name="project-theme")

    files = discover_theme_files(project)

    names = {path.stem for path in files}
    assert names == {"global-theme", "project-theme"}


def test_discover_custom_themes_skips_invalid_and_reports_warning(tmp_path: Path) -> None:
    _write_theme(tmp_path / ".haiku" / "themes" / "good.json", name="good", primary="#123456")
    (tmp_path / ".haiku" / "themes" / "bad.json").write_text("nope", encoding="utf-8")

    themes, warnings = discover_custom_themes(tmp_path)

    assert [t.name for t in themes] == ["good"]
    assert len(warnings) == 1
    assert "bad.json" in warnings[0]


def test_discover_custom_themes_project_overrides_global(tmp_path: Path, monkeypatch) -> None:
    import coding_agent.modes.interactive.theme as theme_module

    home = tmp_path / "home"
    project = tmp_path / "project"
    monkeypatch.setattr(theme_module, "GLOBAL_THEMES_DIR", home / ".haiku" / "themes")
    _write_theme(
        home / ".haiku" / "themes" / "shared.json", name="shared", primary="#global"
    )
    _write_theme(
        project / ".haiku" / "themes" / "shared.json", name="shared", primary="#project"
    )

    themes, _warnings = discover_custom_themes(project)

    assert len(themes) == 1
    assert themes[0].primary == "#project"
