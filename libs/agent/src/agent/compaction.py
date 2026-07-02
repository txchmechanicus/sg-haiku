from __future__ import annotations

from dataclasses import dataclass

from upstream.models import AssistantMessage, Message, TextContent, ToolCall, UserMessage
from upstream.providers.base import ModelProvider

from agent.entries import EntryRef

SUMMARY_SYSTEM_PROMPT = (
    "You summarize coding-agent conversation history so it can be dropped from context "
    "without losing what matters. Write a concise summary covering: the user's goal, "
    "progress made so far, key decisions, and outstanding next steps. Plain text, no preamble."
)

_CHARS_PER_TOKEN = 4


def estimate_tokens(message: Message) -> int:
    if isinstance(message, AssistantMessage) and message.usage.totalTokens:
        return message.usage.totalTokens
    return len(_render_message_text(message)) // _CHARS_PER_TOKEN


def estimate_context_tokens(entries: list[EntryRef]) -> int:
    return sum(estimate_tokens(entry.message) for entry in entries)


def should_compact(context_tokens: int, context_window: int, reserve_tokens: int) -> bool:
    return context_tokens > (context_window - reserve_tokens)


def find_cut_index(entries: list[EntryRef], keep_recent_tokens: int) -> int:
    """Return the index to cut at, landing on a user-message boundary.

    Never splits an assistant tool call from its ToolResultMessage: a turn always
    starts with a UserMessage, so the cut point is advanced forward to the next one.
    """
    if not entries:
        return 0

    total = 0
    cut = None
    for index in range(len(entries) - 1, -1, -1):
        total += estimate_tokens(entries[index].message)
        if total > keep_recent_tokens:
            cut = index
            break
    if cut is None:
        return 0

    forward = cut
    while forward < len(entries) and entries[forward].message.role != "user":
        forward += 1
    if forward < len(entries):
        return forward

    backward = cut
    while backward > 0 and entries[backward].message.role != "user":
        backward -= 1
    return backward


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int


async def summarize(
    provider: ModelProvider,
    messages: list[Message],
    *,
    previous_summary: str | None,
) -> str:
    prompt = _build_summary_prompt(messages, previous_summary)
    response = await provider.complete(
        [UserMessage(content=prompt)],
        [],
        system_prompt=SUMMARY_SYSTEM_PROMPT,
    )
    return response.content.strip() or "Compacted conversation (no summary text returned)."


async def compact(
    provider: ModelProvider,
    entries: list[EntryRef],
    *,
    previous_summary: str | None = None,
    keep_recent_tokens: int,
) -> CompactionResult:
    tokens_before = estimate_context_tokens(entries)
    if not entries:
        return CompactionResult(
            summary=previous_summary or "", first_kept_entry_id="", tokens_before=tokens_before
        )

    cut_index = find_cut_index(entries, keep_recent_tokens)
    to_summarize = [entry.message for entry in entries[:cut_index]]
    if to_summarize:
        summary = await summarize(provider, to_summarize, previous_summary=previous_summary)
    else:
        summary = previous_summary or ""
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=entries[cut_index].id,
        tokens_before=tokens_before,
    )


def _build_summary_prompt(messages: list[Message], previous_summary: str | None) -> str:
    parts = []
    if previous_summary:
        parts.append(f"Previous summary:\n{previous_summary}")
    transcript = "\n".join(_render_message_text(message) for message in messages)
    parts.append(f"Conversation transcript:\n{transcript}")
    parts.append(
        "Summarize this conversation concisely, covering: goal, progress so far, "
        "key decisions made, and next steps."
    )
    return "\n\n".join(parts)


def _render_message_text(message: Message) -> str:
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return f"User: {message.content}"
        return "User: " + "".join(
            part.text for part in message.content if isinstance(part, TextContent)
        )
    if isinstance(message, AssistantMessage):
        text = "".join(part.text for part in message.content if isinstance(part, TextContent))
        calls = [part for part in message.content if isinstance(part, ToolCall)]
        call_text = "".join(f"\n[tool call: {call.name}({call.arguments})]" for call in calls)
        return f"Assistant: {text}{call_text}"
    text = "".join(part.text for part in message.content if isinstance(part, TextContent))
    return f"Tool result ({message.toolName}): {text}"
