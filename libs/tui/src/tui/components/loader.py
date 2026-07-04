from __future__ import annotations

from tui.component import Component

_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Loader(Component):
    """An animated spinner. Advances one frame per `invalidate()` call, so an
    external ticker (e.g. an asyncio task calling `invalidate()` + a TUI
    `request_render()` on an interval) drives the animation.
    """

    def __init__(self, *, label: str = "Thinking") -> None:
        self.label = label
        self._frame = 0

    def invalidate(self) -> None:
        self._frame = (self._frame + 1) % len(_FRAMES)

    def render(self, width: int) -> list[str]:
        line = f"{_FRAMES[self._frame]} {self.label}…"
        return [line[:width]]
