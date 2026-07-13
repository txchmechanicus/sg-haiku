from __future__ import annotations

import asyncio
import contextlib
import difflib
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from agent.core import ToolCallContext
from upstream.models import ImageContent, TextContent
from upstream.types import AgentToolResult

from coding_agent.tools.core import Tool, ToolRegistry
from coding_agent.tools.images import read_image_content, sniff_image_mime_type

MAX_OUTPUT_CHARS = 20_000
MAX_READ_LINES = 2_000
MAX_LINE_CHARS = 2_000
DEFAULT_BASH_TIMEOUT_SECONDS = 30
DEFAULT_GREP_LIMIT = 100
DEFAULT_FIND_LIMIT = 1000
DEFAULT_LS_LIMIT = 500


def default_registry(cwd: Path) -> ToolRegistry:
    return create_all_tools(cwd)


def create_all_tools(cwd: Path) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        _ls_tool(cwd),
        _read_tool(cwd),
        _bash_tool(cwd),
        _write_tool(cwd),
        _edit_tool(cwd),
        _grep_tool(cwd),
        _find_tool(cwd),
    ):
        registry.register(tool)
    return registry


def create_coding_tools(cwd: Path) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (_read_tool(cwd), _bash_tool(cwd), _edit_tool(cwd), _write_tool(cwd)):
        registry.register(tool)
    return registry


def create_read_only_tools(cwd: Path) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (_read_tool(cwd), _grep_tool(cwd), _find_tool(cwd), _ls_tool(cwd)):
        registry.register(tool)
    return registry


def _ls_tool(cwd: Path) -> Tool:
    root = cwd.resolve()

    async def handler(arguments: dict[str, Any], _ctx: ToolCallContext) -> AgentToolResult:
        path = _resolve_inside_root(root, str(arguments.get("path") or "."))
        limit = _positive_int(arguments.get("limit"), DEFAULT_LS_LIMIT, "limit")
        if not path.exists():
            raise ValueError(f"Path not found: {path}")
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        entries = sorted(path.iterdir(), key=lambda entry: entry.name.lower())
        lines: list[str] = []
        entry_limit_reached = False
        for entry in entries:
            if len(lines) >= limit:
                entry_limit_reached = True
                break
            lines.append(f"{entry.name}/" if entry.is_dir() else entry.name)

        if not lines:
            return AgentToolResult.text("(empty directory)"), False
        text, truncation = _truncate_text("\n".join(lines), max_lines=None)
        details: dict[str, object] = {}
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
            details["entryLimitReached"] = limit
        if truncation["truncated"]:
            notices.append(f"{MAX_OUTPUT_CHARS} chars limit reached")
            details["truncation"] = truncation
        if notices:
            text += f"\n\n[{'. '.join(notices)}]"
        return AgentToolResult.text(text, details=details or None), False

    return Tool(
        name="ls",
        label="ls",
        description=(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix "
            "for directories. Includes dotfiles."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list", "default": "."},
                "limit": {
                    "type": "number",
                    "description": "Maximum number of entries",
                    "default": DEFAULT_LS_LIMIT,
                },
            },
            "additionalProperties": False,
        },
        handler=handler,
    )


def _read_tool(cwd: Path) -> Tool:
    root = cwd.resolve()

    async def handler(arguments: dict[str, Any], _ctx: ToolCallContext) -> AgentToolResult:
        path = _resolve_inside_root(root, str(arguments.get("path") or ""))
        offset = _positive_int(arguments.get("offset"), 1, "offset")
        limit = arguments.get("limit")
        limit_value = _positive_int(limit, 0, "limit") if limit is not None else None
        if not path.exists():
            raise ValueError(f"File does not exist: {_display_path(root, path)}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {_display_path(root, path)}")

        mime_type = sniff_image_mime_type(path)
        if mime_type is not None:
            data, final_mime_type = read_image_content(path, mime_type)
            note = f"[Image: {_display_path(root, path)}]"
            return (
                AgentToolResult(
                    content=[
                        TextContent(text=note),
                        ImageContent(data=data, mimeType=final_mime_type),
                    ]
                ),
                False,
            )

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        start = max(0, offset - 1)
        if start >= len(lines):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(lines)} lines total)")

        if limit_value is not None:
            selected_lines = lines[start : start + limit_value]
        else:
            selected_lines = lines[start:]
        selected = "\n".join(selected_lines)
        output, truncation = _truncate_text(selected, max_lines=MAX_READ_LINES)
        details = {"truncation": truncation} if truncation["truncated"] else None

        output_lines = output.split("\n") if output else []
        end_line = start + max(len(output_lines), 1)
        if truncation["truncated"]:
            output += (
                f"\n\n[Showing lines {offset}-{end_line} of {len(lines)}. "
                f"Use offset={end_line + 1} to continue.]"
            )
        elif limit_value is not None and start + len(selected_lines) < len(lines):
            next_offset = start + len(selected_lines) + 1
            remaining = len(lines) - (start + len(selected_lines))
            output += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"

        return AgentToolResult.text(output, details=details), False

    return Tool(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. For text files, output is truncated. "
            "Use offset/limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {
                    "type": "number",
                    "description": "1-indexed line number to start reading",
                },
                "limit": {"type": "number", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _bash_tool(cwd: Path) -> Tool:
    root = cwd.resolve()

    async def handler(arguments: dict[str, Any], ctx: ToolCallContext) -> AgentToolResult:
        command = str(arguments.get("command") or "")
        timeout = _positive_int(arguments.get("timeout"), DEFAULT_BASH_TIMEOUT_SECONDS, "timeout")
        if not command.strip():
            raise ValueError("command must be a non-empty string")

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        # Races the process against both the timeout and the turn's abort signal (if any)
        # so a Ctrl-C during a long-running command actually kills the shell process
        # instead of leaving it running in the background while its result is discarded.
        communicate_task = asyncio.ensure_future(process.communicate())
        waiters: list[asyncio.Future[Any]] = [communicate_task]
        abort_wait_task: asyncio.Future[Any] | None = None
        if ctx.abort_event is not None:
            abort_wait_task = asyncio.ensure_future(ctx.abort_event.wait())
            waiters.append(abort_wait_task)

        try:
            done, _pending = await asyncio.wait(
                waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            if abort_wait_task is not None and not abort_wait_task.done():
                abort_wait_task.cancel()

        if communicate_task in done:
            stdout, stderr = communicate_task.result()
        else:
            communicate_task.cancel()
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            if abort_wait_task is not None and abort_wait_task in done:
                return AgentToolResult.text("Operation aborted"), True
            raise TimeoutError(f"Command timed out after {timeout} seconds")

        output = (stdout + stderr).decode(errors="replace")
        output = output or "(no output)"
        text, truncation = _truncate_text(output, max_lines=None, keep_tail=True)
        details: dict[str, Any] = {}
        if truncation["truncated"]:
            details["truncation"] = truncation
        if process.returncode:
            text += f"\n\nExit code: {process.returncode}"
            details["exitCode"] = process.returncode
            return AgentToolResult.text(text, details=details or None), True
        return AgentToolResult.text(text, details=details or None), False

    return Tool(
        name="bash",
        label="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _write_tool(cwd: Path) -> Tool:
    root = cwd.resolve()

    async def handler(arguments: dict[str, Any], _ctx: ToolCallContext) -> AgentToolResult:
        path_arg = str(arguments.get("path") or "")
        path = _resolve_inside_root(root, path_arg)
        content = str(arguments.get("content") or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return AgentToolResult.text(f"Successfully wrote {len(content)} bytes to {path_arg}"), False

    return Tool(
        name="write",
        label="write",
        description="Write content to a file. Creates parent directories.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _edit_tool(cwd: Path) -> Tool:
    root = cwd.resolve()

    async def handler(arguments: dict[str, Any], _ctx: ToolCallContext) -> AgentToolResult:
        path_arg = str(arguments.get("path") or "")
        path = _resolve_inside_root(root, path_arg)
        edits = _normalize_edits(arguments)
        if not edits:
            raise ValueError(
                "Edit tool input is invalid. edits must contain at least one replacement."
            )
        if not path.exists():
            raise ValueError(f"Could not edit file: {path_arg}. File does not exist.")
        if not path.is_file():
            raise ValueError(f"Could not edit file: {path_arg}. Path is not a file.")

        original = path.read_text(encoding="utf-8", errors="replace")
        updated = original
        for edit in edits:
            old_text = edit["oldText"]
            new_text = edit["newText"]
            count = original.count(old_text)
            if count == 0:
                raise ValueError(f"oldText was not found in {path_arg}.")
            if count > 1:
                raise ValueError(f"oldText must be unique in {path_arg}.")
            if updated.count(old_text) != 1:
                raise ValueError("edits must not overlap or depend on earlier edits.")
            updated = updated.replace(old_text, new_text, 1)

        path.write_text(updated, encoding="utf-8")
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=f"{path_arg} (before)",
                tofile=f"{path_arg} (after)",
                lineterm="",
            )
        )
        patch_lines = list(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=path_arg,
                tofile=path_arg,
                lineterm="",
            )
        )
        first_changed = _first_changed_line(original, updated)
        return AgentToolResult.text(
            f"Successfully replaced {len(edits)} block(s) in {path_arg}.",
            details={
                "diff": "\n".join(diff_lines),
                "patch": "\n".join(patch_lines),
                "firstChangedLine": first_changed,
            },
        ), False

    return Tool(
        name="edit",
        label="edit",
        description="Edit a single file using exact text replacement.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _grep_tool(cwd: Path) -> Tool:
    root = cwd.resolve()

    async def handler(arguments: dict[str, Any], _ctx: ToolCallContext) -> AgentToolResult:
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            raise ValueError("pattern must be a non-empty string.")
        base = _resolve_inside_root(root, str(arguments.get("path") or "."))
        glob = arguments.get("glob")
        ignore_case = bool(arguments.get("ignoreCase") or False)
        literal = bool(arguments.get("literal") or False)
        context = _positive_int(arguments.get("context"), 0, "context")
        limit = _positive_int(arguments.get("limit"), DEFAULT_GREP_LIMIT, "limit")
        if not base.exists():
            raise ValueError(f"Path not found: {base}")

        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(re.escape(pattern) if literal else pattern, flags)
        files = [base] if base.is_file() else _iter_files(base)
        matches: list[str] = []
        match_limit_reached = False
        for file_path in files:
            rel = _display_path(root if base.is_file() else base, file_path)
            if glob and not fnmatch.fnmatch(rel, str(glob)):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for index, line in enumerate(lines):
                if not regex.search(line):
                    continue
                if len(matches) >= limit:
                    match_limit_reached = True
                    break
                start = max(0, index - context)
                end = min(len(lines), index + context + 1)
                for current in range(start, end):
                    marker = ":" if current == index else "-"
                    matches.append(
                        f"{rel}{marker}{current + 1}{marker} {_truncate_line(lines[current])}"
                    )
                if len(matches) >= limit:
                    match_limit_reached = True
                    break
            if match_limit_reached:
                break

        if not matches:
            return AgentToolResult.text("No matches found"), False
        text, truncation = _truncate_text("\n".join(matches), max_lines=None)
        details: dict[str, object] = {}
        notices: list[str] = []
        if match_limit_reached:
            notices.append(f"{limit} matches limit reached. Use limit={limit * 2} for more")
            details["matchLimitReached"] = limit
        if truncation["truncated"]:
            notices.append(f"{MAX_OUTPUT_CHARS} chars limit reached")
            details["truncation"] = truncation
        if notices:
            text += f"\n\n[{'. '.join(notices)}]"
        return AgentToolResult.text(text, details=details or None), False

    return Tool(
        name="grep",
        label="grep",
        description=(
            "Search file contents for a pattern. Returns matching lines with paths and numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern"},
                "path": {"type": "string", "description": "Directory or file to search"},
                "glob": {"type": "string", "description": "Filter files by glob pattern"},
                "ignoreCase": {"type": "boolean", "description": "Case-insensitive search"},
                "literal": {"type": "boolean", "description": "Treat pattern as literal"},
                "context": {
                    "type": "number",
                    "description": "Context lines before and after matches",
                },
                "limit": {"type": "number", "description": "Maximum number of matches"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _find_tool(cwd: Path) -> Tool:
    root = cwd.resolve()

    async def handler(arguments: dict[str, Any], _ctx: ToolCallContext) -> AgentToolResult:
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            raise ValueError("pattern must be a non-empty string.")
        base = _resolve_inside_root(root, str(arguments.get("path") or "."))
        limit = _positive_int(arguments.get("limit"), DEFAULT_FIND_LIMIT, "limit")
        if not base.exists():
            raise ValueError(f"Path not found: {base}")
        if not base.is_dir():
            raise ValueError(f"Path is not a directory: {base}")

        matches: list[str] = []
        result_limit_reached = False
        for item in sorted(base.rglob("*"), key=lambda entry: str(entry).lower()):
            rel = _display_path(base, item)
            if not fnmatch.fnmatch(rel, pattern):
                continue
            if len(matches) >= limit:
                result_limit_reached = True
                break
            matches.append(f"{rel}/" if item.is_dir() else rel)
        if not matches:
            return AgentToolResult.text("No files found matching pattern"), False
        text, truncation = _truncate_text("\n".join(matches), max_lines=None)
        details: dict[str, object] = {}
        notices: list[str] = []
        if result_limit_reached:
            notices.append(f"{limit} results limit reached")
            details["resultLimitReached"] = limit
        if truncation["truncated"]:
            notices.append(f"{MAX_OUTPUT_CHARS} chars limit reached")
            details["truncation"] = truncation
        if notices:
            text += f"\n\n[{'. '.join(notices)}]"
        return AgentToolResult.text(text, details=details or None), False

    return Tool(
        name="find",
        label="find",
        description=(
            "Search for files by glob pattern. Returns matching paths relative to the "
            "search directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files"},
                "path": {"type": "string", "description": "Directory to search"},
                "limit": {"type": "number", "description": "Maximum number of results"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _normalize_edits(arguments: dict[str, Any]) -> list[dict[str, str]]:
    edits = arguments.get("edits")
    if isinstance(edits, str):
        import json

        edits = json.loads(edits)
    normalized = list(edits) if isinstance(edits, list) else []
    if isinstance(arguments.get("oldText"), str) and isinstance(arguments.get("newText"), str):
        normalized.append({"oldText": arguments["oldText"], "newText": arguments["newText"]})
    if isinstance(arguments.get("old"), str) and isinstance(arguments.get("new"), str):
        normalized.append({"oldText": arguments["old"], "newText": arguments["new"]})
    return [
        {"oldText": str(edit.get("oldText") or ""), "newText": str(edit.get("newText") or "")}
        for edit in normalized
    ]


def _resolve_inside_root(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes workspace: {value}") from None
    return path


def _display_path(root: Path, path: Path) -> str:
    with contextlib.suppress(ValueError):
        return str(path.relative_to(root))
    return str(path)


def _positive_int(value: Any, default: int, name: str) -> int:
    if value is None:
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if result < 1 and name != "context":
        raise ValueError(f"{name} must be >= 1")
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _truncate_text(
    value: str,
    *,
    max_lines: int | None,
    keep_tail: bool = False,
) -> tuple[str, dict[str, object]]:
    lines = value.splitlines()
    truncated_by_lines = max_lines is not None and len(lines) > max_lines
    if truncated_by_lines:
        lines = lines[-max_lines:] if keep_tail else lines[:max_lines]
    text = "\n".join(lines) if truncated_by_lines else value
    truncated_by_chars = len(text) > MAX_OUTPUT_CHARS
    if truncated_by_chars:
        text = text[-MAX_OUTPUT_CHARS:] if keep_tail else text[:MAX_OUTPUT_CHARS]
    return text, {
        "truncated": truncated_by_lines or truncated_by_chars,
        "truncatedBy": "lines" if truncated_by_lines else "bytes" if truncated_by_chars else None,
        "totalLines": len(value.splitlines()),
        "outputLines": len(text.splitlines()),
        "maxLines": max_lines,
        "maxBytes": MAX_OUTPUT_CHARS,
    }


def _truncate_line(value: str) -> str:
    return value if len(value) <= MAX_LINE_CHARS else value[:MAX_LINE_CHARS] + "..."


def _iter_files(path: Path) -> list[Path]:
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item))


def _first_changed_line(before: str, after: str) -> int | None:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    for index, (old, new) in enumerate(zip(before_lines, after_lines, strict=False), start=1):
        if old != new:
            return index
    if len(before_lines) != len(after_lines):
        return min(len(before_lines), len(after_lines)) + 1
    return None
