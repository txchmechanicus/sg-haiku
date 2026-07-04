from __future__ import annotations

from abc import ABC, abstractmethod

# Zero-width APC (Application Program Command) escape sequence. A focused
# component emits this at the point in its rendered output where the hardware
# cursor should sit; TUI finds and strips it before writing to the terminal.
CURSOR_MARKER = "\x1b_pi:c\x07"


class Component(ABC):
    """Base type for anything that can be rendered into the TUI's line buffer."""

    wants_key_release: bool = False

    @abstractmethod
    def render(self, width: int) -> list[str]:
        """Render this component to a list of lines for the given viewport width."""

    def handle_input(self, data: str) -> None:  # noqa: B027
        """Handle raw input when this component has focus. No-op by default."""

    def invalidate(self) -> None:  # noqa: B027
        """Drop any cached render state. No-op by default."""


class Focusable(Component):
    """A component that can receive focus and display a hardware cursor.

    When `focused` is True, `render()` should emit `CURSOR_MARKER` at the
    desired cursor position within its output.
    """

    def __init__(self) -> None:
        self.focused: bool = False
