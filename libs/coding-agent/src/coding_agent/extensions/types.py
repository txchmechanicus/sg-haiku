"""Types for sg-haiku's extension/hook system.

The 10 pure-notification events (`agent_start`, `agent_end`, `turn_start`, `turn_end`,
`message_start`, `message_update`, `message_end`, `tool_execution_start`,
`tool_execution_update`, `tool_execution_end`) are dispatched using the *existing*
`agent.events.AgentEvent` straight from `Agent.run()`'s stream, rather than a parallel set of
duplicate dataclasses — Python's `on()` doesn't need a distinct type per event name the way a
statically-typed overload-based API would. Likewise `tool_call`/`tool_result` reuse
`agent.core`'s `ToolCall`/`ToolCallHookResult`/`AgentToolResult`/`ToolResultHookResult` types
directly, and `before_provider_request`/`before_agent_start` reuse
`agent.core.ProviderRequestPayload`/`BeforeAgentStartResult`. Only genuinely new event shapes
(session lifecycle, model/thinking selection, context) get their own dataclasses here.

See `/PLAN.md` (untracked) for the extension-system surface intentionally not implemented yet.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel
from upstream.models import Message

from coding_agent.tools.core import Tool

if TYPE_CHECKING:
    from coding_agent.extensions.runner import ExtensionRunner

Handler = Callable[..., Awaitable[object] | object]


@dataclass(frozen=True)
class SourceInfo:
    """Where an extension came from, for error reporting."""

    path: str
    resolved_path: str


@dataclass(frozen=True)
class RegisteredTool:
    tool: Tool
    source_info: SourceInfo


@dataclass
class Extension:
    """A loaded extension: its registered handlers and tools. Deliberately minus any
    TUI-only fields (message/entry renderers, commands, flags, shortcuts — see
    `/PLAN.md`)."""

    path: str
    resolved_path: str
    source_info: SourceInfo
    handlers: dict[str, list[Handler]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadExtensionsResult:
    extensions: list[Extension]
    errors: list[ExtensionLoadError]


@dataclass(frozen=True)
class ExtensionLoadError:
    path: str
    error: str


@dataclass(frozen=True)
class ExtensionError:
    """Reported via the runner's error listeners when a handler throws (for event types
    whose dispatch is fail-open — see runner.py for the `tool_call` exception)."""

    extension_path: str
    event: str
    error: str


ExtensionErrorListener = Callable[[ExtensionError], None]


class SessionManagerProtocol(Protocol):
    """The read-only slice of `agent.sessions.SessionManager` extensions may observe.
    sg-haiku has no separate read-only wrapper type; extensions are simply handed the live
    manager and expected not to mutate it."""

    def header(self) -> dict[str, object]: ...


class ExtensionContext:
    """Passed to every handler and to registered tools' `execute()`: a single context type
    shared by both, with live property access (delegates to the runner at access time)
    rather than a frozen snapshot, so state changes mid-run (model, system prompt) are
    reflected without re-creating the context.

    No staleness guard yet (no extension-reload feature exists in sg-haiku — see `/PLAN.md`),
    and no `ui`/`hasUI`/`mode` (no TUI framework — see `/PLAN.md`).
    """

    def __init__(self, runner: ExtensionRunner) -> None:
        self._runner = runner

    @property
    def cwd(self) -> Path:
        return self._runner.cwd

    @property
    def session_manager(self) -> SessionManagerProtocol:
        return self._runner.session_manager

    def is_idle(self) -> bool:
        return self._runner.is_idle()

    def get_system_prompt(self) -> str:
        return self._runner.get_system_prompt()

    def request_compact(self) -> None:
        self._runner.request_compact()


# --- Resources discovery -----------------------------------------------------
#
# `promptPaths`/`themePaths` are not implemented: prompt templates are a single-file
# mechanism in sg-haiku (shape mismatch with a path-list model, see `/PLAN.md`), and there
# is no theme system at all. Only `skillPaths` is wired.

ResourcesDiscoverReason = Literal["startup", "reload"]


class ResourcesDiscoverEvent(BaseModel):
    type: Literal["resources_discover"] = "resources_discover"
    cwd: str
    reason: ResourcesDiscoverReason = "startup"


class ResourcesDiscoverResult(BaseModel):
    skillPaths: list[str] | None = None


@dataclass(frozen=True)
class ResourcesDiscoverCollected:
    """Aggregated across every extension's `resources_discover` handler, in load order."""

    skill_paths: tuple[Path, ...] = ()


# --- Session lifecycle ------------------------------------------------------
#
# Unlike the internal-only dataclasses above, these use camelCase pydantic `BaseModel` field
# names, matching the rest of this codebase's convention for stable external-facing contracts,
# and making them golden-fixture-testable the same way `agent.events.AgentEvent` is (see
# tests/contracts/).

SessionStartReason = Literal["new", "resume", "fork"]
SessionShutdownReason = Literal["new", "resume", "fork", "quit"]


class SessionStartEvent(BaseModel):
    type: Literal["session_start"] = "session_start"
    reason: SessionStartReason = "new"


class SessionShutdownEvent(BaseModel):
    type: Literal["session_shutdown"] = "session_shutdown"
    reason: SessionShutdownReason = "quit"


CompactReason = Literal["manual", "threshold"]


class SessionBeforeCompactEvent(BaseModel):
    type: Literal["session_before_compact"] = "session_before_compact"
    reason: CompactReason
    previousSummary: str | None = None


class SessionBeforeCompactResult(BaseModel):
    """`cancel=True` skips compaction entirely. A non-`None` `summary` is passed through as
    `agent.compaction.compact(..., provided_summary=...)`, which already sets
    `CompactionResult.from_hook=True` with no new plumbing needed."""

    cancel: bool = False
    summary: str | None = None


class SessionCompactEvent(BaseModel):
    type: Literal["session_compact"] = "session_compact"
    summary: str
    firstKeptEntryId: str
    tokensBefore: int
    fromExtension: bool


# --- Model / thinking level --------------------------------------------------


class ModelSelectEvent(BaseModel):
    type: Literal["model_select"] = "model_select"
    provider: str
    modelId: str


class ThinkingLevelSelectEvent(BaseModel):
    type: Literal["thinking_level_select"] = "thinking_level_select"
    thinkingLevel: str


# --- Context ------------------------------------------------------------


class ContextEventResult(BaseModel):
    messages: list[Message] | None = None


# --- message_end --------------------------------------------------------


@dataclass(frozen=True)
class MessageEndEventResult:
    """A non-`None` `message` replaces the finalized message; it must keep the same role
    (`agent.extensions.runner.ExtensionRunner.emit_message_end` rejects a role change)."""

    message: Message | None = None
