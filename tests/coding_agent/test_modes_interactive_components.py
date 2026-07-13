from __future__ import annotations

from coding_agent.modes.interactive.components import ToolExecutionComponent


def test_tool_execution_shows_pending_call() -> None:
    component = ToolExecutionComponent("echo", {"x": 1})
    assert str(component.render()) == "⏺ echo(x=1)"


def test_tool_execution_quotes_string_args() -> None:
    component = ToolExecutionComponent("ls", {"path": "."})
    assert str(component.render()) == '⏺ ls(path=".")'


def test_tool_execution_finish_ok_has_no_failure_note() -> None:
    component = ToolExecutionComponent("echo", {})
    component.finish(is_error=False)
    assert str(component.render()) == "⏺ echo()"
    assert component.has_class("ok")


def test_tool_execution_finish_error_appends_failure_note() -> None:
    component = ToolExecutionComponent("echo", {})
    component.finish(is_error=True)
    assert str(component.render()).endswith("— failed")
    assert component.has_class("error")


def test_tool_execution_shows_result_preview() -> None:
    component = ToolExecutionComponent("ls", {"path": "."})
    component.finish(is_error=False, result={"content": [{"type": "text", "text": "a.py\nb.py"}]})

    assert str(component.render()) == '⏺ ls(path=".")\n  ⎿  a.py\n     b.py'


def test_tool_execution_truncates_long_result_preview() -> None:
    component = ToolExecutionComponent("read", {"path": "big.txt"})
    lines = [f"line {i}" for i in range(10)]
    result = {"content": [{"type": "text", "text": "\n".join(lines)}]}
    component.finish(is_error=False, result=result)

    rendered = str(component.render())
    assert rendered.count("\n") == 4  # bullet line + 3 kept preview lines + an ellipsis line
    assert rendered.endswith("…")
