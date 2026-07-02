from __future__ import annotations

from dataclasses import dataclass, field

from upstream.models import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from upstream.providers.base import ModelProvider

from agent.entries import EntryRef

_READ_TOOL_NAMES = {"read"}
_WRITE_TOOL_NAMES = {"write", "edit"}

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
class CompactionDetails:
    """Files touched during a summarized span, mirroring Pi's CompactionEntry.details."""

    readFiles: list[str] = field(default_factory=list)
    modifiedFiles: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {"readFiles": self.readFiles, "modifiedFiles": self.modifiedFiles}


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: CompactionDetails | None = None


def extract_file_ops(entries: list[EntryRef]) -> CompactionDetails:
    """Derive read/modified file paths from read/write/edit tool calls and their results."""
    read_files: list[str] = []
    modified_files: list[str] = []
    seen_read: set[str] = set()
    seen_modified: set[str] = set()
    pending_calls: dict[str, ToolCall] = {}

    for entry in entries:
        message = entry.message
        if isinstance(message, AssistantMessage):
            for part in message.content:
                if isinstance(part, ToolCall) and (
                    part.name in _READ_TOOL_NAMES or part.name in _WRITE_TOOL_NAMES
                ):
                    pending_calls[part.id] = part
        elif isinstance(message, ToolResultMessage):
            call = pending_calls.pop(message.toolCallId, None)
            if call is None or message.isError:
                continue
            path = call.arguments.get("path")
            if not isinstance(path, str) or not path:
                continue
            if call.name in _READ_TOOL_NAMES:
                if path not in seen_read:
                    seen_read.add(path)
                    read_files.append(path)
            else:
                if path not in seen_modified:
                    seen_modified.add(path)
                    modified_files.append(path)

    return CompactionDetails(readFiles=read_files, modifiedFiles=modified_files)


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
    summarized_entries = entries[:cut_index]
    to_summarize = [entry.message for entry in summarized_entries]
    if to_summarize:
        summary = await summarize(provider, to_summarize, previous_summary=previous_summary)
    else:
        summary = previous_summary or ""

    details = extract_file_ops(summarized_entries)
    if not details.readFiles and not details.modifiedFiles:
        details = None

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=entries[cut_index].id,
        tokens_before=tokens_before,
        details=details,
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
