from __future__ import annotations

from tui.component import Component
from tui.container import Container


class _Line(Component):
    def __init__(self, text: str) -> None:
        self.text = text
        self.invalidated = False

    def render(self, width: int) -> list[str]:
        return [self.text]

    def invalidate(self) -> None:
        self.invalidated = True


def test_container_render_concatenates_children() -> None:
    container = Container()
    container.add(_Line("a"))
    container.add(_Line("b"))
    assert container.render(80) == ["a", "b"]


def test_container_remove_and_clear() -> None:
    container = Container()
    child = _Line("a")
    container.add(child)
    container.remove(child)
    assert container.render(80) == []

    container.add(_Line("x"))
    container.clear()
    assert container.render(80) == []


def test_container_invalidate_propagates_to_children() -> None:
    container = Container()
    child = _Line("a")
    container.add(child)
    container.invalidate()
    assert child.invalidated is True
