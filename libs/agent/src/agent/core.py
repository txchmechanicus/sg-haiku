from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from upstream.models import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from upstream.providers import ModelProvider
from upstream.types import AgentToolResult, ThinkingLevel, ToolSpec

from agent.events import AgentEvent

SYSTEM_PROMPT = "You are Haiku, a concise coding agent. Use tools when they help answer the user."


@dataclass
class ProviderRequestPayload:
    """The `before_provider_request` hook payload: a handler receives this and returns a
    (possibly different) instance, which is used verbatim for the actual provider call."""

    messages: list[Message]
    specs: list[ToolSpec]
    system_prompt: str


@dataclass(frozen=True)
class ToolCallHookResult:
    """To patch arguments, mutate `call.arguments` in place before returning rather than
    returning a replacement here."""

    block: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class ToolResultHookResult:
    """Field-patch semantics: omitted (`None`) fields keep whatever the tool execution
    produced."""

    content: list[TextContent | ImageContent] | None = None
    details: object = None
    is_error: bool | None = None


@dataclass(frozen=True)
class BeforeAgentStartResult:
    """Combined result of the `before_agent_start` hook: extra messages to splice in before
    the user prompt, and/or a system prompt override for this run."""

    messages: list[Message] | None = None
    system_prompt: str | None = None


@dataclass(frozen=True)
class AfterProviderResponseInfo:
    """Reported once per completed (non-raising) `ModelProvider.stream()` call.
    `status`/`headers` are always `None`/`{}`: `ModelProvider` doesn't surface
    transport-level response metadata to the agent loop, so this is honest about not having
    it rather than inventing values. `duration_ms` is real wall-clock time spent inside the
    `stream()` call."""

    duration_ms: float
    status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)


BeforeToolCallHook = Callable[[ToolCall], Awaitable[ToolCallHookResult | None]]
AfterToolCallHook = Callable[
    [ToolCall, AgentToolResult, bool], Awaitable[ToolResultHookResult | None]
]
BeforeProviderRequestHook = Callable[[ProviderRequestPayload], Awaitable[ProviderRequestPayload]]
AfterProviderResponseHook = Callable[[AfterProviderResponseInfo], Awaitable[None]]
BeforeAgentStartHook = Callable[
    [str | list[TextContent | ImageContent], str], Awaitable[BeforeAgentStartResult | None]
]
ToolContextProvider = Callable[[], object | None]


@dataclass(frozen=True)
class ToolCallContext:
    """Passed as the second argument to every `ToolExecutor.run()` call.

    `on_update` lets a tool report incremental progress. It is buffered here rather than
    truly concurrent: whatever the handler passes to it is collected, and once the handler
    returns, each buffered value is replayed as a `tool_execution_update` `AgentEvent`, in
    order, right before `tool_execution_end` for that call (see `Agent.run()`). That's a
    deliberate simplification for sg-haiku's headless, line-at-a-time CLI output — there is
    no live-rendering consumer that would benefit from truly interleaved delivery.

    `ext_context` is whatever extra runtime object the caller wants a tool to see. `Agent`
    itself has no opinion on its type (set via `Agent(provide_tool_context=...)`) —
    `coding_agent` supplies its `ExtensionContext` here so custom tools registered by
    extensions (and built-in tools) can read `cwd`/`session_manager`/etc.

    `abort_event` is the same `asyncio.Event` `Agent.abort()` sets and every `ModelProvider`
    already polls between stream chunks — a long-running tool (e.g. a shell command) can
    check `aborted()` periodically and stop early instead of running to completion after the
    user has already cancelled the turn.
    """

    on_update: Callable[[Any], None]
    ext_context: object | None = None
    abort_event: asyncio.Event | None = None

    def aborted(self) -> bool:
        return self.abort_event is not None and self.abort_event.is_set()


class ToolExecutor(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    async def run(self, call: ToolCall, ctx: ToolCallContext) -> tuple[AgentToolResult, bool]: ...

    def execution_mode_for(self, name: str) -> Literal["sequential", "parallel"] | None: ...


class EmptyToolExecutor:
    def specs(self) -> list[ToolSpec]:
        return []

    async def run(self, call: ToolCall, ctx: ToolCallContext) -> tuple[AgentToolResult, bool]:
        return AgentToolResult.text(f"Unknown tool: {call.name}"), True

    def execution_mode_for(self, name: str) -> Literal["sequential", "parallel"] | None:
        return None


def _coerce_user_message(message: str | Message) -> Message:
    return UserMessage(content=message) if isinstance(message, str) else message


class Agent:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        tools: ToolExecutor | None = None,
        cwd: Path | None = None,
        max_tool_iterations: int = 8,
        system_prompt: str = SYSTEM_PROMPT,
        tool_execution_mode: Literal["sequential", "parallel"] = "parallel",
        default_reasoning: ThinkingLevel | None = None,
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
        before_provider_request: BeforeProviderRequestHook | None = None,
        after_provider_response: AfterProviderResponseHook | None = None,
        before_agent_start: BeforeAgentStartHook | None = None,
        provide_tool_context: ToolContextProvider | None = None,
    ) -> None:
        self.provider = provider
        self.cwd = (cwd or Path.cwd()).resolve()
        self.tools = tools or EmptyToolExecutor()
        self.max_tool_iterations = max_tool_iterations
        self.system_prompt = system_prompt
        self.tool_execution_mode = tool_execution_mode
        self.default_reasoning = default_reasoning
        # Hook seams: assignable callbacks (`before_tool_call`/`after_tool_call`,
        # `before_provider_request`/`after_provider_response`, `before_agent_start`). `Agent`
        # itself stays hook-mechanism-agnostic: a single callable each, which `coding_agent`'s
        # extension runner binds to fan out across loaded extensions.
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.before_provider_request = before_provider_request
        self.after_provider_response = after_provider_response
        self.before_agent_start = before_agent_start
        self.provide_tool_context = provide_tool_context
        self._abort_event: asyncio.Event | None = None
        self._steering_queue: list[Message] = []
        self._followup_queue: list[Message] = []

    def abort(self) -> None:
        """Requests cancellation of the currently active `run()` call, if any. Safe to call
        from a concurrently scheduled task while `run()`'s generator is suspended (single
        event loop, no locking needed). A no-op if no run is active."""
        if self._abort_event is not None:
            self._abort_event.set()

    def steer(self, message: str | Message) -> None:
        """Queues a message to be injected before the next provider turn of the active (or
        next) run. Persists across `run()` calls until drained. Safe to call from a
        concurrently scheduled task."""
        self._steering_queue.append(_coerce_user_message(message))

    def follow_up(self, message: str | Message) -> None:
        """Queues a message to be injected only once the run would otherwise stop (no more
        tool calls and no pending steering messages). Persists across `run()` calls until
        drained. Safe to call from a concurrently scheduled task."""
        self._followup_queue.append(_coerce_user_message(message))

    @staticmethod
    def _drain(queue: list[Message]) -> list[Message]:
        drained, queue[:] = list(queue), []
        return drained

    async def _execute_tool(
        self, call: ToolCall
    ) -> tuple[ToolCall, AgentToolResult, bool, list[Any]]:
        if self._abort_event is not None and self._abort_event.is_set():
            return call, AgentToolResult.text("Operation aborted"), True, []

        if self.before_tool_call is not None:
            # The runner's `tool_call` dispatch has no try/except of its own (an extension
            # bug should be visible), but the call site that invokes it converts a thrown
            # exception into a blocked/failed tool call rather than crashing the whole
            # agent run.
            try:
                veto = await self.before_tool_call(call)
            except Exception as exc:  # noqa: BLE001 - converted into a tool-visible error.
                reason = f"Extension failed, blocking execution: {exc}"
                return call, AgentToolResult.text(reason), True, []
            if veto is not None and veto.block:
                reason = veto.reason or f"Tool call to {call.name!r} blocked by extension."
                return call, AgentToolResult.text(reason), True, []

        updates: list[Any] = []
        ctx = ToolCallContext(
            on_update=updates.append,
            ext_context=self.provide_tool_context() if self.provide_tool_context else None,
            abort_event=self._abort_event,
        )
        try:
            result, is_error = await self.tools.run(call, ctx)
        except Exception as exc:  # noqa: BLE001 - converted into a tool-visible error.
            result, is_error = AgentToolResult.text(str(exc)), True

        if self.after_tool_call is not None:
            patch = await self.after_tool_call(call, result, is_error)
            if patch is not None:
                result = AgentToolResult(
                    content=patch.content if patch.content is not None else result.content,
                    details=patch.details if patch.details is not None else result.details,
                    terminate=result.terminate,
                )
                is_error = patch.is_error if patch.is_error is not None else is_error

        return call, result, is_error, updates

    def _effective_execution_mode(self, name: str) -> Literal["sequential", "parallel"]:
        return self.tools.execution_mode_for(name) or self.tool_execution_mode

    @staticmethod
    def _all_results_terminate(
        executed: list[tuple[ToolCall, AgentToolResult, bool, list[Any]]],
    ) -> bool:
        """All-must-agree: the batch signals early completion only if every tool result in
        it set `terminate=True` (a single non-terminating result vetoes early completion)."""
        return bool(executed) and all(result.terminate is True for _, result, _, _ in executed)

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[tuple[ToolCall, AgentToolResult, bool, list[Any]]]:
        """Groups consecutive calls sharing an effective "parallel" mode into one
        `asyncio.gather` batch; a "sequential" call always runs alone. Per-tool
        `execution_mode` overrides `self.tool_execution_mode`."""
        executed: list[tuple[ToolCall, AgentToolResult, bool, list[Any]]] = []
        index = 0
        while index < len(tool_calls):
            call = tool_calls[index]
            if self._effective_execution_mode(call.name) == "parallel":
                group = [call]
                index += 1
                while (
                    index < len(tool_calls)
                    and self._effective_execution_mode(tool_calls[index].name) == "parallel"
                ):
                    group.append(tool_calls[index])
                    index += 1
                executed.extend(await asyncio.gather(*[self._execute_tool(c) for c in group]))
            else:
                executed.append(await self._execute_tool(call))
                index += 1
        return executed

    async def run(
        self,
        prompt: str | list[TextContent | ImageContent],
        *,
        initial_messages: list[Message] | None = None,
        system_prompt: str | None = None,
        use_tools: bool = True,
        reasoning: ThinkingLevel | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        messages: list[Message] = list(initial_messages or [])
        effective_system_prompt = system_prompt or self.system_prompt
        effective_reasoning = reasoning if reasoning is not None else self.default_reasoning
        self._abort_event = asyncio.Event()

        if self.before_agent_start is not None:
            preface = await self.before_agent_start(prompt, effective_system_prompt)
            if preface is not None:
                if preface.messages:
                    messages.extend(preface.messages)
                if preface.system_prompt is not None:
                    effective_system_prompt = preface.system_prompt

        user_message = UserMessage(content=prompt)
        messages.append(user_message)
        specs = self.tools.specs() if use_tools else []

        yield AgentEvent.agent_start()
        yield AgentEvent.message_start(user_message)
        yield AgentEvent.message_end(user_message)

        try:
            for _ in range(self.max_tool_iterations + 1):
                yield AgentEvent.turn_start()

                for steering_message in self._drain(self._steering_queue):
                    messages.append(steering_message)
                    yield AgentEvent.message_start(steering_message)
                    yield AgentEvent.message_end(steering_message)

                assistant_message: AssistantMessage | None = None
                request_payload = ProviderRequestPayload(
                    messages=messages,
                    specs=specs,
                    system_prompt=effective_system_prompt,
                )
                if self.before_provider_request is not None:
                    request_payload = await self.before_provider_request(request_payload)
                stream_started_at = time.monotonic()
                async for assistant_event in self.provider.stream(
                    request_payload.messages,
                    request_payload.specs,
                    system_prompt=request_payload.system_prompt,
                    reasoning=effective_reasoning,
                    abort_event=self._abort_event,
                ):
                    current_message = (
                        assistant_event.partial or assistant_event.message or assistant_event.error
                    )
                    if current_message is not None:
                        assistant_message = current_message
                        if assistant_event.type == "start":
                            yield AgentEvent.message_start(current_message)
                        yield AgentEvent.message_update(current_message, assistant_event)

                if self.after_provider_response is not None:
                    duration_ms = (time.monotonic() - stream_started_at) * 1000
                    await self.after_provider_response(
                        AfterProviderResponseInfo(duration_ms=duration_ms)
                    )

                if assistant_message is None:
                    assistant_message = AssistantMessage(
                        content=[TextContent(text="Provider produced no assistant message.")],
                        stopReason="error",
                        errorMessage="Provider produced no assistant message.",
                    )

                messages.append(assistant_message)
                yield AgentEvent.message_end(assistant_message)

                if assistant_message.stopReason in ("error", "aborted"):
                    yield AgentEvent.turn_end(assistant_message, [])
                    yield AgentEvent.agent_end(messages)
                    return

                tool_calls = [
                    part for part in assistant_message.content if isinstance(part, ToolCall)
                ]
                if not tool_calls:
                    injected = self._drain(self._steering_queue) or self._drain(
                        self._followup_queue
                    )
                    if injected:
                        yield AgentEvent.turn_end(assistant_message, [])
                        for injected_message in injected:
                            messages.append(injected_message)
                            yield AgentEvent.message_start(injected_message)
                            yield AgentEvent.message_end(injected_message)
                        continue
                    yield AgentEvent.turn_end(assistant_message, [])
                    yield AgentEvent.agent_end(messages)
                    return

                if not use_tools:
                    error_message = ToolResultMessage(
                        toolCallId=tool_calls[0].id,
                        toolName=tool_calls[0].name,
                        content=[
                            TextContent(text="Model requested tools, but tools are disabled.")
                        ],
                        isError=True,
                    )
                    messages.append(error_message)
                    yield AgentEvent.message_start(error_message)
                    yield AgentEvent.message_end(error_message)
                    yield AgentEvent.turn_end(assistant_message, [error_message])
                    yield AgentEvent.agent_end(messages)
                    return

                for call in tool_calls:
                    yield AgentEvent.tool_execution_start(call.id, call.name, call.arguments)

                executed = await self._execute_tool_calls(tool_calls)

                tool_results: list[ToolResultMessage] = []
                for call, result, is_error, updates in executed:
                    for update in updates:
                        yield AgentEvent.tool_execution_update(
                            call.id, call.name, call.arguments, update
                        )
                    yield AgentEvent.tool_execution_end(
                        call.id,
                        call.name,
                        call.arguments,
                        result.model_dump(mode="json", exclude_none=True),
                        is_error,
                    )
                    result_message = ToolResultMessage(
                        toolCallId=call.id,
                        toolName=call.name,
                        content=result.content,
                        details=result.details,
                        isError=is_error,
                    )
                    messages.append(result_message)
                    tool_results.append(result_message)
                    yield AgentEvent.message_start(result_message)
                    yield AgentEvent.message_end(result_message)

                yield AgentEvent.turn_end(assistant_message, tool_results)

                if self._abort_event.is_set():
                    yield AgentEvent.agent_end(messages)
                    return

                if self._all_results_terminate(executed):
                    yield AgentEvent.agent_end(messages)
                    return

            error_message = AssistantMessage(
                content=[
                    TextContent(text=f"Stopped after {self.max_tool_iterations} tool iterations.")
                ],
                stopReason="error",
                errorMessage=f"Stopped after {self.max_tool_iterations} tool iterations.",
            )
            messages.append(error_message)
            yield AgentEvent.message_start(error_message)
            yield AgentEvent.message_end(error_message)
            yield AgentEvent.agent_end(messages)
        except Exception as exc:  # noqa: BLE001 - guarantees a terminal agent_end.
            fallback_message = AssistantMessage(
                content=[TextContent(text=str(exc))],
                stopReason="error",
                errorMessage=str(exc),
            )
            messages.append(fallback_message)
            yield AgentEvent.message_start(fallback_message)
            yield AgentEvent.message_end(fallback_message)
            yield AgentEvent.turn_end(fallback_message, [])
            yield AgentEvent.agent_end(messages)
