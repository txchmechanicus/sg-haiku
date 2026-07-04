from __future__ import annotations

from tui.component import Component


class Container(Component):
    """A component that renders a vertical stack of child components."""

    def __init__(self) -> None:
        self.children: list[Component] = []

    def add(self, component: Component) -> None:
        self.children.append(component)

    def remove(self, component: Component) -> None:
        self.children.remove(component)

    def clear(self) -> None:
        self.children = []

    def invalidate(self) -> None:
        for child in self.children:
            child.invalidate()

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines
