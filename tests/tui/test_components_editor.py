from __future__ import annotations

from tui.component import CURSOR_MARKER
from tui.components.editor import LineEditor


def test_editor_appends_typed_characters() -> None:
    editor = LineEditor(prompt="> ")
    editor.handle_input("h")
    editor.handle_input("i")
    assert editor.text == "hi"
    assert editor.cursor == 2


def test_editor_backspace_and_delete() -> None:
    editor = LineEditor(prompt="> ")
    editor.set_text("hello")
    editor.cursor = 5
    editor.handle_input("\x7f")
    assert editor.text == "hell"

    editor.cursor = 0
    editor.handle_input("\x1b[3~")
    assert editor.text == "ell"


def test_editor_arrow_navigation() -> None:
    editor = LineEditor(prompt="> ")
    editor.set_text("abc")
    editor.cursor = 3
    editor.handle_input("\x1b[D")
    assert editor.cursor == 2
    editor.handle_input("\x1b[C")
    assert editor.cursor == 3


def test_editor_enter_submits_and_clears() -> None:
    submitted: list[str] = []
    editor = LineEditor(prompt="> ", on_submit=submitted.append)
    editor.set_text("hello")
    editor.handle_input("\r")

    assert submitted == ["hello"]
    assert editor.text == ""
    assert editor.cursor == 0


def test_editor_render_shows_cursor_marker_only_when_focused() -> None:
    editor = LineEditor(prompt="> ")
    editor.set_text("hi")
    assert CURSOR_MARKER not in editor.render(80)[0]

    editor.focused = True
    line = editor.render(80)[0]
    assert CURSOR_MARKER in line
    assert line.replace(CURSOR_MARKER, "") == "> hi"
