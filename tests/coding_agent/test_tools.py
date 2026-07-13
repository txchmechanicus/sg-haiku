from __future__ import annotations

from pathlib import Path

import pytest
from coding_agent.tools import (
    create_all_tools,
    create_coding_tools,
    create_read_only_tools,
    default_registry,
)
from upstream import ToolCall
from upstream.types import AgentToolResult


def test_registry_execution_mode_for(tmp_path: Path) -> None:
    from coding_agent.tools.core import Tool, ToolRegistry

    async def handler(_args, _ctx):  # noqa: ANN001
        raise NotImplementedError

    registry = ToolRegistry()
    registry.register(Tool(name="plain", description="", parameters={}, handler=handler))
    registry.register(
        Tool(
            name="seq",
            description="",
            parameters={},
            handler=handler,
            execution_mode="sequential",
        )
    )

    assert registry.execution_mode_for("plain") is None
    assert registry.execution_mode_for("seq") == "sequential"
    assert registry.execution_mode_for("missing") is None


@pytest.mark.asyncio
async def test_run_applies_prepare_arguments_before_handler(tmp_path: Path) -> None:
    from coding_agent.tools.core import Tool, ToolRegistry

    seen_args = {}

    async def handler(args, _ctx):  # noqa: ANN001
        seen_args.update(args)
        return AgentToolResult.text("ok"), False

    def prepare(args: dict) -> dict:
        return {**args, "path": args["path"].strip()}

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="coerce",
            description="",
            parameters={},
            handler=handler,
            prepare_arguments=prepare,
        )
    )

    result, is_error = await registry.run(
        ToolCall(id="1", name="coerce", arguments={"path": "  a.txt  "})
    )

    assert is_error is False
    assert seen_args == {"path": "a.txt"}


@pytest.mark.asyncio
async def test_run_reports_prepare_arguments_failure_as_tool_error(tmp_path: Path) -> None:
    from coding_agent.tools.core import Tool, ToolRegistry

    async def handler(_args, _ctx):  # noqa: ANN001
        raise AssertionError("handler should not run")

    def prepare(_args: dict) -> dict:
        raise ValueError("bad shape")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="coerce",
            description="",
            parameters={},
            handler=handler,
            prepare_arguments=prepare,
        )
    )

    result, is_error = await registry.run(ToolCall(id="1", name="coerce", arguments={}))

    assert is_error is True
    assert "bad shape" in result.content[0].text


def test_registry_prompt_snippets_and_guidelines(tmp_path: Path) -> None:
    from coding_agent.tools.core import Tool, ToolRegistry

    async def handler(_args, _ctx):  # noqa: ANN001
        raise NotImplementedError

    registry = ToolRegistry()
    registry.register(Tool(name="plain", description="", parameters={}, handler=handler))
    registry.register(
        Tool(
            name="custom",
            description="",
            parameters={},
            handler=handler,
            prompt_snippet="custom: does a thing",
            prompt_guidelines=("Call custom before finishing.", "Never call custom twice."),
        )
    )

    assert registry.prompt_snippets() == ["custom: does a thing"]
    assert registry.prompt_guidelines() == [
        "Call custom before finishing.",
        "Never call custom twice.",
    ]


def test_registry_names_and_filtering(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    assert registry.names() == ["ls", "read", "bash", "write", "edit", "grep", "find"]
    assert registry.filtered(include={"read", "ls"}).names() == ["ls", "read"]
    assert registry.filtered(exclude={"bash", "write"}).names() == [
        "ls",
        "read",
        "edit",
        "grep",
        "find",
    ]


def test_registry_filtering_rejects_unknown_tool(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    with pytest.raises(ValueError, match="Unknown tool name: missing"):
        registry.filtered(include={"missing"})


def test_tool_group_factories(tmp_path: Path) -> None:
    assert create_all_tools(tmp_path).names() == [
        "ls",
        "read",
        "bash",
        "write",
        "edit",
        "grep",
        "find",
    ]
    assert create_coding_tools(tmp_path).names() == ["read", "bash", "edit", "write"]
    assert create_read_only_tools(tmp_path).names() == ["read", "grep", "find", "ls"]


def test_tool_schemas_use_contract_argument_names(tmp_path: Path) -> None:
    specs = {spec.name: spec.parameters for spec in default_registry(tmp_path).specs()}

    assert set(specs["read"]["properties"]) == {"path", "offset", "limit"}
    assert set(specs["bash"]["properties"]) == {"command", "timeout"}
    assert set(specs["edit"]["properties"]) == {"path", "edits"}
    assert set(specs["grep"]["properties"]) == {
        "pattern",
        "path",
        "glob",
        "ignoreCase",
        "literal",
        "context",
        "limit",
    }
    assert set(specs["find"]["properties"]) == {"pattern", "path", "limit"}
    assert set(specs["ls"]["properties"]) == {"path", "limit"}


@pytest.mark.asyncio
async def test_ls_lists_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(ToolCall(id="1", name="ls", arguments={"path": "."}))

    assert is_error is False
    assert result.content[0].text.splitlines() == ["a.txt", "dir/"]


@pytest.mark.asyncio
async def test_ls_limit_sets_details(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="ls", arguments={"path": ".", "limit": 1})
    )

    assert is_error is False
    assert result.content[0].text.startswith("a.txt")
    assert result.details["entryLimitReached"] == 1


@pytest.mark.asyncio
async def test_read_reads_file(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="read", arguments={"path": "README.md"})
    )

    assert is_error is False
    assert result.content[0].text == "hello"


@pytest.mark.asyncio
async def test_read_offset_and_limit_adds_continuation_notice(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("one\ntwo\nthree\nfour", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="read",
            arguments={"path": "README.md", "offset": 2, "limit": 2},
        )
    )

    assert is_error is False
    assert result.content[0].text == (
        "two\nthree\n\n[1 more lines in file. Use offset=4 to continue.]"
    )


@pytest.mark.asyncio
async def test_read_offset_beyond_eof_is_error(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("one", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="read", arguments={"path": "README.md", "offset": 5})
    )

    assert is_error is True
    assert "Offset 5 is beyond end of file" in result.content[0].text


@pytest.mark.asyncio
async def test_read_blocks_workspace_escape(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="read", arguments={"path": "../secret"})
    )

    assert is_error is True
    assert "escapes workspace" in result.content[0].text


@pytest.mark.asyncio
async def test_read_returns_image_content_for_png(tmp_path: Path) -> None:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (10, 10), color="red").save(buffer, format="PNG")
    (tmp_path / "pic.png").write_bytes(buffer.getvalue())
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="read", arguments={"path": "pic.png"})
    )

    assert is_error is False
    assert result.content[0].type == "text"
    assert "pic.png" in result.content[0].text
    assert result.content[1].type == "image"
    assert result.content[1].mimeType == "image/png"
    assert result.content[1].data


@pytest.mark.asyncio
async def test_read_resizes_oversized_image(tmp_path: Path) -> None:
    import base64
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (3000, 2500), color="blue").save(buffer, format="PNG")
    (tmp_path / "big.png").write_bytes(buffer.getvalue())
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="read", arguments={"path": "big.png"})
    )

    assert is_error is False
    decoded = Image.open(BytesIO(base64.b64decode(result.content[1].data)))
    assert decoded.width <= 2000
    assert decoded.height <= 2000


@pytest.mark.asyncio
async def test_bash_runs_command(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="bash", arguments={"command": "printf hi"})
    )

    assert is_error is False
    assert result.content[0].text == "hi"


@pytest.mark.asyncio
async def test_bash_combines_stderr_and_reports_nonzero(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="bash",
            arguments={"command": "printf out; printf err >&2; exit 3"},
        )
    )

    assert is_error is True
    assert "outerr" in result.content[0].text
    assert "Exit code: 3" in result.content[0].text


@pytest.mark.asyncio
async def test_bash_timeout(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="bash", arguments={"command": "sleep 2", "timeout": 1})
    )

    assert is_error is True
    assert "Command timed out after 1 seconds" in result.content[0].text


@pytest.mark.asyncio
async def test_bash_stops_promptly_when_aborted(tmp_path: Path) -> None:
    import asyncio
    import time

    from agent.core import ToolCallContext

    registry = default_registry(tmp_path)
    abort_event = asyncio.Event()

    async def abort_soon() -> None:
        await asyncio.sleep(0.3)
        abort_event.set()

    asyncio.ensure_future(abort_soon())
    ctx = ToolCallContext(on_update=lambda _update: None, abort_event=abort_event)

    start = time.monotonic()
    result, is_error = await registry.run(
        ToolCall(id="1", name="bash", arguments={"command": "sleep 30", "timeout": 30}),
        ctx,
    )
    elapsed = time.monotonic() - start

    assert is_error is True
    assert result.content[0].text == "Operation aborted"
    assert elapsed < 2  # nowhere near the 30s timeout/sleep duration

    # The subprocess must actually be killed, not just abandoned.
    await asyncio.sleep(0.3)
    proc = await asyncio.create_subprocess_exec(
        "pgrep", "-f", "sleep 30", stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    assert out.decode().strip() == ""


@pytest.mark.asyncio
async def test_write_creates_parent_directories_and_file(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="write",
            arguments={"path": "nested/example.txt", "content": "hello"},
        )
    )

    assert is_error is False
    assert (tmp_path / "nested" / "example.txt").read_text(encoding="utf-8") == "hello"
    assert result.content[0].text == "Successfully wrote 5 bytes to nested/example.txt"


@pytest.mark.asyncio
async def test_write_blocks_workspace_escape(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="write", arguments={"path": "../x.txt", "content": "x"})
    )

    assert is_error is True
    assert "escapes workspace" in result.content[0].text


@pytest.mark.asyncio
async def test_edit_replaces_single_match(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="edit",
            arguments={"path": "a.txt", "oldText": "world", "newText": "haiku"},
        )
    )

    assert is_error is False
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello haiku"
    assert result.content[0].text == "Successfully replaced 1 block(s) in a.txt."
    assert "diff" in result.details
    assert "patch" in result.details


@pytest.mark.asyncio
async def test_edit_requires_unique_old_text(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x x", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="edit",
            arguments={"path": "a.txt", "edits": [{"oldText": "x", "newText": "y"}]},
        )
    )

    assert is_error is True
    assert "oldText must be unique" in result.content[0].text


@pytest.mark.asyncio
async def test_edit_multiple_edits(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one two three", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="edit",
            arguments={
                "path": "a.txt",
                "edits": [
                    {"oldText": "one", "newText": "1"},
                    {"oldText": "three", "newText": "3"},
                ],
            },
        )
    )

    assert is_error is False
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "1 two 3"
    assert result.content[0].text == "Successfully replaced 2 block(s) in a.txt."


@pytest.mark.asyncio
async def test_edit_reports_no_match(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("abc", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="edit",
            arguments={"path": "a.txt", "oldText": "missing", "newText": "value"},
        )
    )

    assert is_error is True
    assert "oldText was not found" in result.content[0].text


@pytest.mark.asyncio
async def test_grep_finds_matches_case_insensitive_and_limits(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Alpha\nbeta\nALPHA\n", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="grep",
            arguments={
                "pattern": "alpha",
                "ignoreCase": True,
                "limit": 1,
            },
        )
    )

    assert is_error is False
    assert result.content[0].text.startswith("a.txt:1: Alpha")


@pytest.mark.asyncio
async def test_grep_literal_glob_and_context(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("before\nalpha.beta\nafter\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("alphaXbeta\n", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(
            id="1",
            name="grep",
            arguments={
                "pattern": "alpha.beta",
                "literal": True,
                "glob": "*.txt",
                "context": 1,
            },
        )
    )

    assert is_error is False
    assert result.content[0].text.splitlines() == [
        "a.txt-1- before",
        "a.txt:2: alpha.beta",
        "a.txt-3- after",
    ]


@pytest.mark.asyncio
async def test_grep_blocks_workspace_escape(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="grep", arguments={"pattern": "x", "path": "../"})
    )

    assert is_error is True
    assert "escapes workspace" in result.content[0].text


@pytest.mark.asyncio
async def test_find_matches_glob_and_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="find", arguments={"pattern": "*", "limit": 3})
    )

    assert is_error is False
    assert result.content[0].text.splitlines() == ["README.md", "src/", "src/app.py"]


@pytest.mark.asyncio
async def test_find_blocks_workspace_escape(tmp_path: Path) -> None:
    registry = default_registry(tmp_path)

    result, is_error = await registry.run(
        ToolCall(id="1", name="find", arguments={"path": "../", "pattern": "**/*"})
    )

    assert is_error is True
    assert "escapes workspace" in result.content[0].text
