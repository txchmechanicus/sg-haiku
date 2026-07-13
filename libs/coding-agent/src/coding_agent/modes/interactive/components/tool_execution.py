from __future__ import annotations

from typing import Any

from textual.widgets import Static

_BULLET = "⏺"
_MAX_VALUE_LEN = 60
_MAX_PREVIEW_LINES = 3
_MAX_PREVIEW_CHARS = 200


def _format_args(args: Any) -> str:
    if not isinstance(args, dict):
        return str(args)
    parts = []
    for key, value in args.items():
        text = value if isinstance(value, str) else repr(value)
        text = str(text)
        if len(text) > _MAX_VALUE_LEN:
            text = text[: _MAX_VALUE_LEN - 1] + "…"
        parts.append(f'{key}="{text}"' if isinstance(value, str) else f"{key}={text}")
    return ", ".join(parts)


def result_text(result: Any) -> str:
    """Extracts a plain-text preview from an `AgentToolResult`-shaped payload (a dumped
    dict with a `content` list of `{"type": "text", "text": ...}` parts, as delivered by
    `tool_execution_end`) or from a list of `TextContent`-like objects (as replayed from a
    session's `ToolResultMessage.content`)."""
    if isinstance(result, dict):
        parts = result.get("content") or []
    elif isinstance(result, list):
        parts = result
    else:
        return ""
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
        else:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def _format_preview(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    truncated_chars = len(text) > _MAX_PREVIEW_CHARS
    if truncated_chars:
        text = text[:_MAX_PREVIEW_CHARS]
    all_lines = text.splitlines() or [text]
    truncated_lines = len(all_lines) > _MAX_PREVIEW_LINES
    lines = all_lines[:_MAX_PREVIEW_LINES]
    if truncated_chars or truncated_lines:
        lines.append("…")
    return "\n".join(
        [f"  ⎿  {lines[0]}", *(f"     {line}" for line in lines[1:])]
    )


class ToolExecutionComponent(Static):
    """Renders a single tool call as a compact bullet line, plus a short indented preview
    of its result once finished — dim while running, green on success, red on error."""

    DEFAULT_CSS = """
    ToolExecutionComponent { color: $text-muted; }
    ToolExecutionComponent.ok { color: $success; }
    ToolExecutionComponent.error { color: $error; }
    """

    def __init__(self, tool_name: str, args: Any) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.args = args
        self.finished = False
        self.is_error = False
        self.result_preview = ""
        self._refresh_text()

    def finish(self, *, is_error: bool, result: Any = None) -> None:
        self.finished = True
        self.is_error = is_error
        self.result_preview = result_text(result)
        self.add_class("error" if is_error else "ok")
        self._refresh_text()

    def _refresh_text(self) -> None:
        line = f"{_BULLET} {self.tool_name}({_format_args(self.args)})"
        if self.finished:
            if self.is_error:
                line += " — failed"
            preview = _format_preview(self.result_preview)
            if preview:
                line += "\n" + preview
        self.update(line)
