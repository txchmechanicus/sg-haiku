from __future__ import annotations

import pytest
from agent.compaction import (
    compact,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_index,
    should_compact,
)
from upstream import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from upstream.providers import MockProvider


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

    assert estimate_context_tokens([one, two]) == estimate_tokens(one) + estimate_tokens(two)


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
    messages = [
        UserMessage(content="a" * 400),
        AssistantMessage(
            content=[ToolCall(id="call-1", name="ls", arguments={})],
            stopReason="toolUse",
        ),
        ToolResultMessage(
            toolCallId="call-1", toolName="ls", content=[TextContent(text="x" * 400)], isError=False
        ),
        UserMessage(content="b" * 400),
        AssistantMessage(content=[TextContent(text="c" * 400)]),
    ]

    cut = find_cut_index(messages, keep_recent_tokens=10)

    assert messages[cut].role == "user"
    assert cut in (0, 3)


def test_find_cut_index_keeps_everything_when_budget_is_large() -> None:
    messages = [UserMessage(content="hi"), AssistantMessage(content=[TextContent(text="hey")])]

    assert find_cut_index(messages, keep_recent_tokens=10_000) == 0


def test_find_cut_index_empty_messages() -> None:
    assert find_cut_index([], keep_recent_tokens=100) == 0


@pytest.mark.asyncio
async def test_compact_summarizes_and_cuts(monkeypatch) -> None:
    messages = [
        UserMessage(content="a" * 400),
        AssistantMessage(content=[TextContent(text="b" * 400)]),
        UserMessage(content="c" * 400),
        AssistantMessage(content=[TextContent(text="d" * 400)]),
    ]
    provider = MockProvider()

    result = await compact(provider, messages, keep_recent_tokens=10)

    assert result.cut_index > 0
    assert result.summary
    assert result.tokens_before == estimate_context_tokens(messages)


@pytest.mark.asyncio
async def test_compact_reuses_previous_summary_when_nothing_to_cut() -> None:
    messages = [UserMessage(content="hi")]
    provider = MockProvider()

    result = await compact(
        provider, messages, previous_summary="earlier summary", keep_recent_tokens=10_000
    )

    assert result.cut_index == 0
    assert result.summary == "earlier summary"
