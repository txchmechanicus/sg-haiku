from __future__ import annotations

import io

from rich.console import Console

from tui.component import Component


class Text(Component):
    """A word-wrapped block of (optionally styled) text.

    Wrapping and ANSI styling are delegated to `rich`, which is already a
    dependency of this package, rather than hand-rolling terminal styling.
    """

    def __init__(self, text: str = "", *, style: str | None = None) -> None:
        self.text = text
        self.style = style

    def set_text(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        if not self.text:
            return [""]
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=max(width, 1),
            force_terminal=True,
            color_system="standard",
            highlight=False,
            markup=False,
        )
        console.print(self.text, style=self.style, end="")
        rendered = buffer.getvalue()
        return rendered.split("\n") if rendered else [""]
