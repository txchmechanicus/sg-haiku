"""`ExtensionRunner` — dispatches events to loaded extensions' handlers.

Dispatch is **not** uniform across event types — each event type has its own merge policy
rather than a single generic dispatcher (see each `emit_*` method's docstring). All dispatch
is sequential, in extension-load order, then handler-registration order within an extension —
never parallel.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from agent.core import (
    BeforeAgentStartResult,
    ProviderRequestPayload,
    ToolCallHookResult,
    ToolResultHookResult,
)
from upstream.models import Message
from upstream.types import AgentToolResult

from coding_agent.extensions.types import (
    ContextEventResult,
    Extension,
    ExtensionContext,
    ExtensionError,
    ExtensionErrorListener,
    ResourcesDiscoverCollected,
    ResourcesDiscoverEvent,
    ResourcesDiscoverReason,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
    SessionManagerProtocol,
)

if TYPE_CHECKING:
    from upstream.models import ToolCall

    from coding_agent.tools.core import ToolRegistry


class ExtensionRunner:
    def __init__(
        self,
        extensions: list[Extension],
        *,
        cwd: Path,
        session_manager: SessionManagerProtocol,
        get_system_prompt: Callable[[], str] = lambda: "",
        is_idle: Callable[[], bool] = lambda: True,
        request_compact: Callable[[], None] = lambda: None,
        error_listener: ExtensionErrorListener | None = None,
    ) -> None:
        self.extensions = extensions
        self.cwd = cwd
        self.session_manager = session_manager
        self._get_system_prompt = get_system_prompt
        self._is_idle = is_idle
        self._request_compact = request_compact
        self._error_listener = error_listener

    def create_context(self) -> ExtensionContext:
        return ExtensionContext(self)

    def is_idle(self) -> bool:
        return self._is_idle()

    def get_system_prompt(self) -> str:
        return self._get_system_prompt()

    def request_compact(self) -> None:
        self._request_compact()

    def has_handlers(self, event_type: str) -> bool:
        return any(extension.handlers.get(event_type) for extension in self.extensions)

    def register_tools(self, registry: ToolRegistry) -> None:
        for extension in self.extensions:
            for registered in extension.tools.values():
                registry.register(registered.tool)

    def _emit_error(self, extension: Extension, event_type: str, error: Exception) -> None:
        if self._error_listener is not None:
            self._error_listener(
                ExtensionError(extension_path=extension.path, event=event_type, error=str(error))
            )

    async def _call(self, handler: Callable[..., object], /, *args: object) -> object:
        result = handler(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    # -- Pure notifications: no return value used, fail-open (error reported, continue). ----

    async def notify(self, event_type: str, event: object) -> None:
        ctx = self.create_context()
        for extension in self.extensions:
            handlers = extension.handlers.get(event_type)
            if not handlers:
                continue
            for handler in handlers:
                try:
                    await self._call(handler, event, ctx)
                except Exception as exc:  # noqa: BLE001 - per-handler isolation.
                    self._emit_error(extension, event_type, exc)

    # -- message_end: chained full-message replace, role-checked. ---------------------------

    async def emit_message_end(self, message: Message) -> Message | None:
        ctx = self.create_context()
        current = message
        modified = False
        for extension in self.extensions:
            handlers = extension.handlers.get("message_end")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    result = await self._call(handler, current, ctx)
                    if not result or result.message is None:
                        continue
                    if result.message.role != current.role:
                        self._emit_error(
                            extension,
                            "message_end",
                            ValueError(
                                "message_end handlers must return a message with the same role"
                            ),
                        )
                        continue
                    current = result.message
                    modified = True
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(extension, "message_end", exc)
        return current if modified else None

    # -- tool_call: mutate `call.arguments` in place; last non-None result wins for `reason`;
    #    `block=True` short-circuits; NO try/except here — an exception propagates out of
    #    dispatch and is expected to block/fail the tool call (caught by the caller). --------

    async def emit_tool_call(self, call: ToolCall) -> ToolCallHookResult | None:
        ctx = self.create_context()
        result: ToolCallHookResult | None = None
        for extension in self.extensions:
            handlers = extension.handlers.get("tool_call")
            if not handlers:
                continue
            for handler in handlers:
                handler_result = await self._call(handler, call, ctx)
                if handler_result is not None:
                    result = handler_result
                    if result.block:
                        return result
        return result

    # -- tool_result: field-by-field progressive patch; each handler sees the prior handlers'
    #    already-applied patch. ---------------------------------------------------------------

    async def emit_tool_result(
        self, call: ToolCall, result: AgentToolResult, is_error: bool
    ) -> ToolResultHookResult | None:
        ctx = self.create_context()
        current_content = result.content
        current_details = result.details
        current_is_error = is_error
        modified = False
        for extension in self.extensions:
            handlers = extension.handlers.get("tool_result")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    view = AgentToolResult(
                        content=current_content, details=current_details, terminate=result.terminate
                    )
                    handler_result = await self._call(handler, call, view, current_is_error, ctx)
                    if handler_result is None:
                        continue
                    if handler_result.content is not None:
                        current_content = handler_result.content
                        modified = True
                    if handler_result.details is not None:
                        current_details = handler_result.details
                        modified = True
                    if handler_result.is_error is not None:
                        current_is_error = handler_result.is_error
                        modified = True
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(extension, "tool_result", exc)
        if not modified:
            return None
        return ToolResultHookResult(
            content=current_content, details=current_details, is_error=current_is_error
        )

    # -- context: full-array sequential pipeline, last handler's output feeds the next. ------

    async def emit_context(self, messages: list[Message]) -> list[Message]:
        ctx = self.create_context()
        current = list(messages)
        for extension in self.extensions:
            handlers = extension.handlers.get("context")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result: ContextEventResult | None = await self._call(
                        handler, current, ctx
                    )
                    if handler_result is not None and handler_result.messages is not None:
                        current = handler_result.messages
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(extension, "context", exc)
        return current

    # -- before_provider_request: full-payload sequential pipeline, handler's return value is
    #    used verbatim as the next handler's input / the final request. ----------------------

    async def emit_before_provider_request(
        self, payload: ProviderRequestPayload
    ) -> ProviderRequestPayload:
        ctx = self.create_context()
        current = payload
        for extension in self.extensions:
            handlers = extension.handlers.get("before_provider_request")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result = await self._call(handler, current, ctx)
                    if handler_result is not None:
                        current = handler_result
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(extension, "before_provider_request", exc)
        return current

    # -- before_agent_start: messages accumulate across all handlers; system_prompt chains
    #    (each handler sees the prior handlers' override via `ctx.get_system_prompt()`). -----

    async def emit_before_agent_start(
        self, prompt: str, system_prompt: str
    ) -> BeforeAgentStartResult | None:
        current_system_prompt = system_prompt
        ctx = ExtensionContext(self)
        ctx.get_system_prompt = lambda: current_system_prompt  # type: ignore[method-assign]
        messages: list[Message] = []
        modified = False
        for extension in self.extensions:
            handlers = extension.handlers.get("before_agent_start")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result: BeforeAgentStartResult | None = await self._call(
                        handler, prompt, current_system_prompt, ctx
                    )
                    if handler_result is not None:
                        if handler_result.messages:
                            messages.extend(handler_result.messages)
                            modified = True
                        if handler_result.system_prompt is not None:
                            current_system_prompt = handler_result.system_prompt
                            modified = True
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(extension, "before_agent_start", exc)
        if not modified:
            return None
        return BeforeAgentStartResult(
            messages=messages or None,
            system_prompt=current_system_prompt if current_system_prompt != system_prompt else None,
        )

    # -- resources_discover: every handler's skillPaths are collected (not merged/overridden),
    #    in extension-load order, then handler-registration order. -----------------------------

    async def emit_resources_discover(
        self, *, cwd: Path, reason: ResourcesDiscoverReason = "startup"
    ) -> ResourcesDiscoverCollected:
        ctx = self.create_context()
        event = ResourcesDiscoverEvent(cwd=str(cwd), reason=reason)
        skill_paths: list[Path] = []
        for extension in self.extensions:
            handlers = extension.handlers.get("resources_discover")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    result = await self._call(handler, event, ctx)
                    if result is not None and result.skillPaths:
                        skill_paths.extend(Path(path) for path in result.skillPaths)
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(extension, "resources_discover", exc)
        return ResourcesDiscoverCollected(skill_paths=tuple(skill_paths))

    # -- session_before_compact: last non-None result wins; `cancel=True` short-circuits. -----

    async def emit_session_before_compact(
        self, event: SessionBeforeCompactEvent
    ) -> SessionBeforeCompactResult | None:
        ctx = self.create_context()
        result: SessionBeforeCompactResult | None = None
        for extension in self.extensions:
            handlers = extension.handlers.get("session_before_compact")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result = await self._call(handler, event, ctx)
                    if handler_result is not None:
                        result = handler_result
                        if result.cancel:
                            return result
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(extension, "session_before_compact", exc)
        return result
