from __future__ import annotations

import pytest
from agent.compaction import (
    compact,
    estimate_context_tokens,
    estimate_tokens,
    extract_file_ops,
    find_cut_index,
    should_compact,
)
from agent.entries import EntryRef
from upstream import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from upstream.models import Message
from upstream.providers import MockProvider


def _refs(messages: list[Message]) -> list[EntryRef]:
    return [EntryRef(id=f"id-{index}", message=message) for index, message in enumerate(messages)]


def test_estimate_tokens_uses_usage_total_tokens_when_present() -> None:
    message = AssistantMessage(
        content=[TextContent(text="hi")],
        usage=Usage(totalTokens=42),
    )

    assert estimate_tokens(message) == 42


def test_estimate_tokens_falls_back_to_char_heuristic() -> None:
    message = UserMessage(content="a" * 40)

    assert estimate_tokens(message) == estimate_tokens(message)
    assert estimate_tokens(message) > 0
    assert estimate_tokens(UserMessage(content="a" * 400)) > estimate_tokens(
        UserMessage(content="a" * 40)
    )


def test_estimate_context_tokens_sums_messages() -> None:
    one = UserMessage(content="a" * 40)
    two = UserMessage(content="b" * 40)

    assert estimate_context_tokens(_refs([one, two])) == estimate_tokens(one) + estimate_tokens(
        two
    )


@pytest.mark.parametrize(
    ("context_tokens", "context_window", "reserve_tokens", "expected"),
    [
        (100, 200, 50, False),
        (151, 200, 50, True),
        (150, 200, 50, False),
    ],
)
def test_should_compact_threshold(
    context_tokens: int, context_window: int, reserve_tokens: int, expected: bool
) -> None:
    assert should_compact(context_tokens, context_window, reserve_tokens) is expected


def test_find_cut_index_never_splits_tool_call_from_result() -> None:
    entries = _refs(
        [
            UserMessage(content="a" * 400),
            AssistantMessage(
                content=[ToolCall(id="call-1", name="ls", arguments={})],
                stopReason="toolUse",
            ),
            ToolResultMessage(
                toolCallId="call-1",
                toolName="ls",
                content=[TextContent(text="x" * 400)],
                isError=False,
            ),
            UserMessage(content="b" * 400),
            AssistantMessage(content=[TextContent(text="c" * 400)]),
        ]
    )

    cut = find_cut_index(entries, keep_recent_tokens=10)

    assert entries[cut].message.role == "user"
    assert cut in (0, 3)


def test_find_cut_index_keeps_everything_when_budget_is_large() -> None:
    entries = _refs(
        [UserMessage(content="hi"), AssistantMessage(content=[TextContent(text="hey")])]
    )

    assert find_cut_index(entries, keep_recent_tokens=10_000) == 0


def test_find_cut_index_empty_messages() -> None:
    assert find_cut_index([], keep_recent_tokens=100) == 0


@pytest.mark.asyncio
async def test_compact_summarizes_and_cuts(monkeypatch) -> None:
    entries = _refs(
        [
            UserMessage(content="a" * 400),
            AssistantMessage(content=[TextContent(text="b" * 400)]),
            UserMessage(content="c" * 400),
            AssistantMessage(content=[TextContent(text="d" * 400)]),
        ]
    )
    provider = MockProvider()

    result = await compact(provider, entries, keep_recent_tokens=10)

    assert result.first_kept_entry_id in {entry.id for entry in entries}
    assert result.first_kept_entry_id != entries[0].id
    assert result.summary
    assert result.tokens_before == estimate_context_tokens(entries)
    assert result.from_hook is False


class _RaisingProvider(MockProvider):
    """Fails if the LLM summarization path is ever invoked."""

    async def stream(self, messages, tools, system_prompt=None):
        raise AssertionError("LLM summarization should not be called with provided_summary")
        yield  # pragma: no cover - unreachable, keeps this an async generator


@pytest.mark.asyncio
async def test_compact_uses_provided_summary_and_sets_from_hook() -> None:
    entries = _refs(
        [
            UserMessage(content="a" * 400),
            AssistantMessage(content=[TextContent(text="b" * 400)]),
            UserMessage(content="c" * 400),
            AssistantMessage(content=[TextContent(text="d" * 400)]),
        ]
    )
    provider = _RaisingProvider()

    result = await compact(
        provider, entries, keep_recent_tokens=10, provided_summary="hook-supplied summary"
    )

    assert result.summary == "hook-supplied summary"
    assert result.from_hook is True


@pytest.mark.asyncio
async def test_compact_provided_summary_with_empty_history() -> None:
    provider = _RaisingProvider()

    result = await compact(
        provider, [], keep_recent_tokens=10, provided_summary="hook-supplied summary"
    )

    assert result.summary == "hook-supplied summary"
    assert result.from_hook is True
    assert result.first_kept_entry_id == ""


@pytest.mark.asyncio
async def test_compact_reuses_previous_summary_when_nothing_to_cut() -> None:
    entries = _refs([UserMessage(content="hi")])
    provider = MockProvider()

    result = await compact(
        provider, entries, previous_summary="earlier summary", keep_recent_tokens=10_000
    )

    assert result.first_kept_entry_id == entries[0].id
    assert result.summary == "earlier summary"
    assert result.from_hook is False


@pytest.mark.asyncio
async def test_compact_handles_empty_history() -> None:
    provider = MockProvider()

    result = await compact(provider, [], previous_summary="earlier summary", keep_recent_tokens=10)

    assert result.first_kept_entry_id == ""
    assert result.summary == "earlier summary"
    assert result.tokens_before == 0
    assert result.from_hook is False


def _read_call_and_result(call_id: str, path: str, *, is_error: bool = False) -> list[Message]:
    return [
        AssistantMessage(
            content=[ToolCall(id=call_id, name="read", arguments={"path": path})],
            stopReason="toolUse",
        ),
        ToolResultMessage(
            toolCallId=call_id,
            toolName="read",
            content=[TextContent(text="contents")],
            isError=is_error,
        ),
    ]


def _write_call_and_result(call_id: str, path: str, *, is_error: bool = False) -> list[Message]:
    return [
        AssistantMessage(
            content=[ToolCall(id=call_id, name="edit", arguments={"path": path})],
            stopReason="toolUse",
        ),
        ToolResultMessage(
            toolCallId=call_id,
            toolName="edit",
            content=[TextContent(text="ok")],
            isError=is_error,
        ),
    ]


def test_extract_file_ops_tracks_reads_and_writes() -> None:
    messages = [
        UserMessage(content="do stuff"),
        *_read_call_and_result("call-1", "a.py"),
        *_write_call_and_result("call-2", "b.py"),
    ]

    details = extract_file_ops(_refs(messages))

    assert details.readFiles == ["a.py"]
    assert details.modifiedFiles == ["b.py"]


def test_extract_file_ops_ignores_errored_tool_results() -> None:
    messages = [*_read_call_and_result("call-1", "a.py", is_error=True)]

    details = extract_file_ops(_refs(messages))

    assert details.readFiles == []
    assert details.modifiedFiles == []


def test_extract_file_ops_deduplicates_paths() -> None:
    messages = [
        *_read_call_and_result("call-1", "a.py"),
        *_read_call_and_result("call-2", "a.py"),
    ]

    details = extract_file_ops(_refs(messages))

    assert details.readFiles == ["a.py"]


@pytest.mark.asyncio
async def test_compact_attaches_details_when_files_are_touched() -> None:
    messages = [
        UserMessage(content="a" * 400),
        *_read_call_and_result("call-1", "a.py"),
        UserMessage(content="c" * 400),
        AssistantMessage(content=[TextContent(text="d" * 400)]),
    ]
    provider = MockProvider()

    result = await compact(provider, _refs(messages), keep_recent_tokens=10)

    assert result.details is not None
    assert result.details.readFiles == ["a.py"]


@pytest.mark.asyncio
async def test_compact_omits_details_when_no_files_are_touched() -> None:
    entries = _refs(
        [
            UserMessage(content="a" * 400),
            AssistantMessage(content=[TextContent(text="b" * 400)]),
            UserMessage(content="c" * 400),
            AssistantMessage(content=[TextContent(text="d" * 400)]),
        ]
    )
    provider = MockProvider()

    result = await compact(provider, entries, keep_recent_tokens=10)

    assert result.details is None
