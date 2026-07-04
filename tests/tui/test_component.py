from __future__ import annotations

from tui.component import CURSOR_MARKER, Component, Focusable


class _DummyComponent(Component):
    def render(self, width: int) -> list[str]:
        return [f"dummy:{width}"]


def test_component_render_returns_lines() -> None:
    component = _DummyComponent()
    assert component.render(10) == ["dummy:10"]


def test_component_default_handle_input_and_invalidate_are_noops() -> None:
    component = _DummyComponent()
    component.handle_input("x")
    component.invalidate()


def test_focusable_starts_unfocused_and_can_be_focused() -> None:
    class _DummyFocusable(Focusable):
        def render(self, width: int) -> list[str]:
            return [CURSOR_MARKER] if self.focused else [""]

    focusable = _DummyFocusable()
    assert focusable.focused is False

    focusable.focused = True
    assert focusable.render(10) == [CURSOR_MARKER]
