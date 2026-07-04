from __future__ import annotations

from tui.component import CURSOR_MARKER
from tui.components.select_list import SelectItem, SelectList


def _items() -> list[SelectItem]:
    return [
        SelectItem(id="a", label="Alpha"),
        SelectItem(id="b", label="Beta", description="second"),
        SelectItem(id="c", label="Gamma"),
    ]


def test_select_list_starts_at_first_item() -> None:
    select_list = SelectList(_items())
    assert select_list.index == 0


def test_select_list_down_and_up_move_selection() -> None:
    select_list = SelectList(_items())
    select_list.handle_input("\x1b[B")  # down
    assert select_list.index == 1
    select_list.handle_input("\x1b[A")  # up
    assert select_list.index == 0


def test_select_list_wraps_around() -> None:
    select_list = SelectList(_items())
    select_list.handle_input("\x1b[A")  # up from 0 wraps to last
    assert select_list.index == 2
    select_list.handle_input("\x1b[B")  # down wraps back to 0
    assert select_list.index == 0


def test_select_list_enter_invokes_on_select_with_current_item() -> None:
    selected: list[SelectItem] = []
    select_list = SelectList(_items(), on_select=selected.append)
    select_list.handle_input("\x1b[B")
    select_list.handle_input("\r")
    assert len(selected) == 1
    assert selected[0].id == "b"


def test_select_list_escape_invokes_on_cancel() -> None:
    cancelled = []
    select_list = SelectList(_items(), on_cancel=lambda: cancelled.append(True))
    select_list.handle_input("\x1b")
    assert cancelled == [True]


def test_select_list_render_highlights_selected_row() -> None:
    select_list = SelectList(_items())
    lines = select_list.render(40)
    assert len(lines) == 3
    assert "\x1b[7m" in lines[0]
    assert "\x1b[7m" not in lines[1]


def test_select_list_render_emits_cursor_marker_only_when_focused() -> None:
    select_list = SelectList(_items())
    assert CURSOR_MARKER not in "".join(select_list.render(40))
    select_list.focused = True
    assert CURSOR_MARKER in select_list.render(40)[0]
