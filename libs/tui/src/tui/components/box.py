from __future__ import annotations

import re

from tui.container import Container

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_width(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


class Box(Container):
    """A bordered container drawn around its children's rendered output."""

    def __init__(self, *, title: str | None = None, padding: int = 0) -> None:
        super().__init__()
        self.title = title
        self.padding = padding

    def render(self, width: int) -> list[str]:
        inner_width = max(width - 2 - 2 * self.padding, 1)
        content_lines = super().render(inner_width)

        pad = " " * self.padding
        body: list[str] = []
        for line in content_lines:
            fill = max(inner_width - _visible_width(line), 0)
            body.append(f"│{pad}{line}{' ' * fill}{pad}│")

        if self.title:
            label = f" {self.title} "
            fill = max(width - 2 - len(label), 0)
            top = f"╭{label}{'─' * fill}╮"
        else:
            top = f"╭{'─' * (width - 2)}╮"
        bottom = f"╰{'─' * (width - 2)}╯"

        return [top, *body, bottom]
