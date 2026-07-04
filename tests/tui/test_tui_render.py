from __future__ import annotations

import asyncio

from tui.component import Component
from tui.tui import TUI

from tests.tui.fake_terminal import FakeTerminal


class _Line(Component):
    def __init__(self, text: str = "") -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        return [self.text]


def test_do_render_writes_full_frame_on_first_render() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    tui.add(_Line("hello"))

    tui._do_render()

    assert "hello" in terminal.output
    assert tui._last_lines == ["hello"]


def test_do_render_skips_write_when_unchanged() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    tui.add(_Line("hello"))

    tui._do_render()
    terminal.written.clear()
    tui._do_render()

    assert terminal.written == []


def test_do_render_only_rewrites_changed_line() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    line_a = _Line("a")
    line_b = _Line("b")
    tui.add(line_a)
    tui.add(line_b)

    tui._do_render()
    terminal.written.clear()

    line_b.text = "changed"
    tui._do_render()

    assert "\r\x1b[Kchanged" in terminal.output
    assert "\r\x1b[Ka" not in terminal.output
    assert tui._last_lines == ["a", "changed"]


async def test_request_render_coalesces_pending_calls() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    tui.add(_Line("x"))

    tui.request_render()
    tui.request_render()
    assert tui._render_pending is True

    await asyncio.sleep(0)

    assert tui._last_lines == ["x"]
    assert tui._render_pending is False
