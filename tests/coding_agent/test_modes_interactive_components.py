from __future__ import annotations

from coding_agent.modes.interactive.components import ToolExecutionComponent


def test_tool_execution_render_never_exceeds_width() -> None:
    component = ToolExecutionComponent("write_file", {"path": "x" * 200, "content": "y" * 200})
    lines = component.render(40)
    assert len(lines) == 1
    assert len(lines[0]) <= 40


def test_tool_execution_render_short_line_is_untouched() -> None:
    component = ToolExecutionComponent("echo", {"x": 1})
    lines = component.render(80)
    assert lines == ["tool: echo({'x': 1})"]


def test_tool_execution_finish_appends_status() -> None:
    ok_component = ToolExecutionComponent("echo", {})
    ok_component.finish(is_error=False)
    assert ok_component.render(80)[0].endswith("-> ok")

    error_component = ToolExecutionComponent("echo", {})
    error_component.finish(is_error=True)
    assert error_component.render(80)[0].endswith("-> error")
