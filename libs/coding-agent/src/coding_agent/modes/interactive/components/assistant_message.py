from __future__ import annotations

from textual.widgets import Static


class AssistantMessageComponent(Static):
    """Renders one assistant turn's accumulated streamed text."""

    DEFAULT_CSS = """
    AssistantMessageComponent.error { color: $error; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self.text = ""

    def update_delta(self, delta: str) -> None:
        self.text += delta
        self.update(self.text)
