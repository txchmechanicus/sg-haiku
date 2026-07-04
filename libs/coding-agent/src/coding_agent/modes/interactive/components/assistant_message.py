from __future__ import annotations

from tui import Text


class AssistantMessageComponent(Text):
    """Renders one assistant turn's accumulated streamed text."""

    def __init__(self) -> None:
        super().__init__(text="")

    def update_delta(self, delta: str) -> None:
        self.text += delta
        self.invalidate()
