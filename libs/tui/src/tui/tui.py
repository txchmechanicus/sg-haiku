from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable

from tui.component import CURSOR_MARKER, Component
from tui.container import Container
from tui.keys import split_keys
from tui.terminal import Terminal

# Returns True if the listener consumed the input (stopping further dispatch).
InputListener = Callable[[str], bool]


class TUI(Container):
    """Owns a `Terminal` and drives a differential re-render loop over its
    tree of `Component` children.

    This is a reduced port of Pi's `TUI` class: it performs real line-level
    diffing against the previously rendered frame (so redraws only touch the
    lines that changed) and hardware-cursor placement via `CURSOR_MARKER`, but
    does not yet implement Pi's overlay/dialog stack or Kitty image
    bookkeeping (deferred to a later phase).
    """

    def __init__(self, terminal: Terminal) -> None:
        super().__init__()
        self.terminal = terminal
        self.focused: Component | None = None
        self._input_listeners: list[InputListener] = []
        self._last_lines: list[str] = []
        self._running = False
        self._render_pending = False
        self._read_task: asyncio.Task[None] | None = None

    def set_focus(self, component: Component | None) -> None:
        if self.focused is not None and hasattr(self.focused, "focused"):
            self.focused.focused = False
        self.focused = component
        if component is not None and hasattr(component, "focused"):
            component.focused = True
        self.request_render()

    def add_input_listener(self, listener: InputListener) -> Callable[[], None]:
        self._input_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._input_listeners:
                self._input_listeners.remove(listener)

        return unsubscribe

    def start(self) -> None:
        self.terminal.enter_raw_mode()
        self._running = True
        self._read_task = asyncio.ensure_future(self._read_loop())
        if hasattr(signal, "SIGWINCH"):
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGWINCH, lambda: self.request_render(force=True)
            )
        self._do_render()

    def stop(self) -> None:
        self._running = False
        if self._read_task is not None:
            self._read_task.cancel()
            self._read_task = None
        if hasattr(signal, "SIGWINCH"):
            asyncio.get_running_loop().remove_signal_handler(signal.SIGWINCH)
        self.terminal.exit_raw_mode()

    async def _read_loop(self) -> None:
        while self._running:
            data = await self.terminal.read()
            self.handle_input(data)

    def handle_input(self, data: bytes) -> None:
        for chunk in split_keys(data):
            text = chunk.decode("utf-8", errors="replace")
            consumed = any(listener(text) for listener in list(self._input_listeners))
            if not consumed and self.focused is not None:
                self.focused.handle_input(text)
        self.request_render()

    def request_render(self, force: bool = False) -> None:
        if force:
            self._last_lines = []
        if self._render_pending:
            return
        self._render_pending = True
        asyncio.get_running_loop().call_soon(self._render_now)

    def _render_now(self) -> None:
        self._render_pending = False
        self._do_render()

    def _do_render(self) -> None:
        width, _height = self.terminal.get_size()
        raw_lines = self.render(width)

        cursor_row: int | None = None
        cursor_col: int | None = None
        new_lines: list[str] = []
        for i, line in enumerate(raw_lines):
            idx = line.find(CURSOR_MARKER)
            if idx != -1:
                cursor_row = i
                cursor_col = idx
                line = line.replace(CURSOR_MARKER, "")
            new_lines.append(line)

        old_lines = self._last_lines
        common = min(len(old_lines), len(new_lines))
        first_diff = 0
        while first_diff < common and old_lines[first_diff] == new_lines[first_diff]:
            first_diff += 1

        if old_lines == new_lines:
            self._last_lines = new_lines
            return

        out: list[str] = []
        if old_lines:
            move_up = len(old_lines) - 1
            if move_up > 0:
                out.append(f"\x1b[{move_up}A")
            out.append("\r")
            if first_diff > 0:
                out.append(f"\x1b[{first_diff}B")

        for i in range(first_diff, len(new_lines)):
            out.append("\r\x1b[K")
            out.append(new_lines[i])
            if i != len(new_lines) - 1:
                out.append("\n")

        if len(old_lines) > len(new_lines):
            extra = len(old_lines) - len(new_lines)
            for _ in range(extra):
                out.append("\n\r\x1b[K")
            out.append(f"\x1b[{extra}A")

        self.terminal.write("".join(out))

        if cursor_row is not None:
            rows_from_bottom = (len(new_lines) - 1) - cursor_row
            cursor_seq = ""
            if rows_from_bottom > 0:
                cursor_seq += f"\x1b[{rows_from_bottom}A"
            cursor_seq += "\r"
            if cursor_col:
                cursor_seq += f"\x1b[{cursor_col}C"
            self.terminal.write(cursor_seq)

        self._last_lines = new_lines
