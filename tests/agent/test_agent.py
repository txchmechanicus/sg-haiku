from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from agent import Agent
from agent.core import (
    BeforeAgentStartResult,
    ProviderRequestPayload,
    ToolCallHookResult,
    ToolResultHookResult,
)
from coding_agent.tools import default_registry
from upstream import (
    AssistantMessage,
    AssistantMessageEvent,
    Message,
    MockProvider,
    ModelProvider,
    ToolCall,
    ToolSpec,
    UserMessage,
)
from upstream.types import AgentToolResult
from upstream.models import TextContent


async def collect(agent: Agent, prompt: str, *, use_tools: bool = True):
    return [event async for event in agent.run(prompt, use_tools=use_tools)]


@pytest.mark.asyncio
async def test_agent_returns_contract_events_without_tools(tmp_path: Path) -> None:
    agent = Agent(provider=MockProvider(), cwd=tmp_path)

    events = await collect(agent, "hello")

    assert events[0].type == "agent_start"
    assert [event.type for event in events].count("message_update") >= 1
    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[0].content[0].text == "Mock response: hello"


@pytest.mark.asyncio
async def test_agent_executes_ls_tool_with_contract_events(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content", encoding="utf-8")
    agent = Agent(provider=MockProvider(), tools=default_registry(tmp_path), cwd=tmp_path)

    events = await collect(agent, "list files")

    assert "tool_execution_start" in [event.type for event in events]
    assert "tool_execution_end" in [event.type for event in events]
    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].content[0].text == "example.txt"


@pytest.mark.asyncio
async def test_agent_reports_disabled_tools_as_tool_result(tmp_path: Path) -> None:
    agent = Agent(provider=MockProvider(), cwd=tmp_path)

    events = await collect(agent, "list files", use_tools=False)

    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[0].content[0].text == "Mock response: list files"


class AlwaysToolProvider(ModelProvider):
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        call = ToolCall(id="call-1", name="missing", arguments={})
        message = AssistantMessage(content=[call], stopReason="toolUse")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=message)
        yield AssistantMessageEvent(
            type="toolcall_end",
            contentIndex=0,
            toolCall=call,
            partial=message,
        )
        yield AssistantMessageEvent(type="done", reason="toolUse", message=message)


@pytest.mark.asyncio
async def test_agent_stops_after_max_tool_iterations(tmp_path: Path) -> None:
    agent = Agent(provider=AlwaysToolProvider(), cwd=tmp_path, max_tool_iterations=1)

    events = await collect(agent, "loop")

    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[-1].stopReason == "error"
    assert "Stopped after 1 tool iterations" in assistant_messages[-1].errorMessage


class CapturingProvider(ModelProvider):
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.system_prompt: str | None = None
        self.reasoning: object | None = None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.messages = list(messages)
        self.system_prompt = system_prompt
        self.reasoning = reasoning
        message = AssistantMessage(content=[], stopReason="stop")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="done", reason="stop", message=message)


@pytest.mark.asyncio
async def test_agent_passes_initial_messages_to_provider(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path)

    events = [
        event
        async for event in agent.run(
            "current",
            initial_messages=[UserMessage(content="previous", timestamp=123)],
        )
    ]

    assert [message.content for message in provider.messages if message.role == "user"] == [
        "previous",
        "current",
    ]
    started_users = [
        event.message.content
        for event in events
        if event.type == "message_start" and getattr(event.message, "role", None) == "user"
    ]
    assert started_users == ["current"]


@pytest.mark.asyncio
async def test_agent_passes_system_prompt_to_provider(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path)

    await collect(agent, "current", use_tools=False)
    assert provider.system_prompt == agent.system_prompt

    _ = [
        event
        async for event in agent.run(
            "current",
            system_prompt="custom system prompt",
            use_tools=False,
        )
    ]
    assert provider.system_prompt == "custom system prompt"


@pytest.mark.asyncio
async def test_agent_passes_reasoning_to_provider(tmp_path: Path) -> None:
    provider = CapturingProvider()

    agent = Agent(provider=provider, cwd=tmp_path)
    await collect(agent, "current", use_tools=False)
    assert provider.reasoning is None

    agent_with_default = Agent(provider=provider, cwd=tmp_path, default_reasoning="medium")
    await collect(agent_with_default, "current", use_tools=False)
    assert provider.reasoning == "medium"

    _ = [
        event
        async for event in agent_with_default.run(
            "current",
            use_tools=False,
            reasoning="high",
        )
    ]
    assert provider.reasoning == "high"


@pytest.mark.asyncio
async def test_before_tool_call_hook_blocks_execution(tmp_path: Path) -> None:
    async def before_tool_call(call: ToolCall) -> ToolCallHookResult | None:
        return ToolCallHookResult(block=True, reason="nope")

    agent = Agent(
        provider=MockProvider(),
        tools=default_registry(tmp_path),
        cwd=tmp_path,
        before_tool_call=before_tool_call,
    )

    events = await collect(agent, "list files")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].isError is True
    assert tool_results[0].content[0].text == "nope"


@pytest.mark.asyncio
async def test_before_tool_call_hook_exception_becomes_tool_error(tmp_path: Path) -> None:
    async def before_tool_call(call: ToolCall) -> ToolCallHookResult | None:
        raise RuntimeError("extension exploded")

    agent = Agent(
        provider=MockProvider(),
        tools=default_registry(tmp_path),
        cwd=tmp_path,
        before_tool_call=before_tool_call,
    )

    events = await collect(agent, "list files")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].isError is True
    assert "extension exploded" in tool_results[0].content[0].text


@pytest.mark.asyncio
async def test_after_tool_call_hook_patches_result(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content", encoding="utf-8")

    async def after_tool_call(
        call: ToolCall, result: AgentToolResult, is_error: bool
    ) -> ToolResultHookResult | None:
        return ToolResultHookResult(details={"patched": True})

    agent = Agent(
        provider=MockProvider(),
        tools=default_registry(tmp_path),
        cwd=tmp_path,
        after_tool_call=after_tool_call,
    )

    events = await collect(agent, "list files")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].details == {"patched": True}
    assert tool_results[0].content[0].text == "example.txt"


@pytest.mark.asyncio
async def test_before_provider_request_hook_patches_payload() -> None:
    provider = CapturingProvider()

    async def before_provider_request(
        payload: ProviderRequestPayload,
    ) -> ProviderRequestPayload:
        payload.system_prompt = "patched by extension"
        return payload

    agent = Agent(provider=provider, before_provider_request=before_provider_request)

    await collect(agent, "hello", use_tools=False)

    assert provider.system_prompt == "patched by extension"


@pytest.mark.asyncio
async def test_before_agent_start_hook_injects_messages_and_system_prompt() -> None:
    provider = CapturingProvider()

    async def before_agent_start(prompt: str, system_prompt: str) -> BeforeAgentStartResult:
        return BeforeAgentStartResult(
            messages=[UserMessage(content="injected", timestamp=1)],
            system_prompt="overridden system prompt",
        )

    agent = Agent(provider=provider, before_agent_start=before_agent_start)

    await collect(agent, "hello", use_tools=False)

    assert provider.system_prompt == "overridden system prompt"
    assert [message.content for message in provider.messages if message.role == "user"] == [
        "injected",
        "hello",
    ]


@pytest.mark.asyncio
async def test_after_provider_response_hook_fires_once_per_completed_stream() -> None:
    from agent.core import AfterProviderResponseInfo

    provider = CapturingProvider()
    infos: list[AfterProviderResponseInfo] = []

    async def after_provider_response(info: AfterProviderResponseInfo) -> None:
        infos.append(info)

    agent = Agent(provider=provider, after_provider_response=after_provider_response)

    await collect(agent, "hello", use_tools=False)

    assert len(infos) == 1
    assert infos[0].status is None
    assert infos[0].headers == {}
    assert infos[0].duration_ms >= 0


@pytest.mark.asyncio
async def test_after_provider_response_hook_is_not_called_when_unset() -> None:
    provider = CapturingProvider()

    agent = Agent(provider=provider)

    events = await collect(agent, "hello", use_tools=False)

    assert events[-1].type == "agent_end"


class UpdatingToolExecutor:
    """A minimal ToolExecutor whose tool reports incremental progress via ctx.on_update and
    reads ctx.ext_context, for exercising the ToolCallContext plumbing end-to-end."""

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name="progress", description="reports progress", parameters={})]

    async def run(self, call: ToolCall, ctx) -> tuple[AgentToolResult, bool]:  # noqa: ANN001
        ctx.on_update({"step": 1})
        ctx.on_update({"step": 2})
        return AgentToolResult.text(f"ext_context={ctx.ext_context!r}"), False

    def execution_mode_for(self, name: str) -> str | None:
        return None


class ProgressToolProvider(ModelProvider):
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        last = messages[-1]
        if last.role == "toolResult":
            message = AssistantMessage(content=[], stopReason="stop")
            yield AssistantMessageEvent(type="start", partial=message)
            yield AssistantMessageEvent(type="done", reason="stop", message=message)
            return
        call = ToolCall(id="call-1", name="progress", arguments={})
        message = AssistantMessage(content=[call], stopReason="toolUse")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(
            type="toolcall_end", contentIndex=0, toolCall=call, partial=message
        )
        yield AssistantMessageEvent(type="done", reason="toolUse", message=message)


@pytest.mark.asyncio
async def test_agent_emits_tool_execution_update_events_before_end(tmp_path: Path) -> None:
    agent = Agent(provider=ProgressToolProvider(), tools=UpdatingToolExecutor(), cwd=tmp_path)

    events = await collect(agent, "go")

    tool_event_types = [
        (event.type, event.partialResult)
        for event in events
        if event.type in {"tool_execution_update", "tool_execution_end"}
    ]
    assert tool_event_types[0] == ("tool_execution_update", {"step": 1})
    assert tool_event_types[1] == ("tool_execution_update", {"step": 2})
    assert tool_event_types[2][0] == "tool_execution_end"


@pytest.mark.asyncio
async def test_agent_passes_ext_context_to_tools(tmp_path: Path) -> None:
    agent = Agent(
        provider=ProgressToolProvider(),
        tools=UpdatingToolExecutor(),
        cwd=tmp_path,
        provide_tool_context=lambda: "my-ext-context",
    )

    events = await collect(agent, "go")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].content[0].text == "ext_context='my-ext-context'"


class MultiToolCallProvider(ModelProvider):
    """Requests three tool calls in a single turn, then finishes on the next turn."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        if messages[-1].role == "toolResult":
            message = AssistantMessage(content=[TextContent(text="done")], stopReason="stop")
            yield AssistantMessageEvent(type="start", partial=message)
            yield AssistantMessageEvent(type="done", reason="stop", message=message)
            return
        calls = [
            ToolCall(id="1", name="a", arguments={}),
            ToolCall(id="2", name="b", arguments={}),
            ToolCall(id="3", name="c", arguments={}),
        ]
        message = AssistantMessage(content=calls, stopReason="toolUse")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="done", reason="toolUse", message=message)


class RecordingToolExecutor:
    """Records `start:<name>`/`end:<name>` around each call, yielding control via
    `asyncio.sleep(0)` so concurrently-gathered calls interleave observably."""

    def __init__(self, modes: dict[str, str | None]) -> None:
        self._modes = modes
        self.log: list[str] = []

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name=name, description="", parameters={}) for name in self._modes]

    async def run(self, call: ToolCall, ctx) -> tuple[AgentToolResult, bool]:  # noqa: ANN001
        self.log.append(f"start:{call.name}")
        await asyncio.sleep(0)
        self.log.append(f"end:{call.name}")
        return AgentToolResult.text("ok"), False

    def execution_mode_for(self, name: str) -> str | None:
        return self._modes.get(name)


@pytest.mark.asyncio
async def test_consecutive_parallel_tools_interleave(tmp_path: Path) -> None:
    tools = RecordingToolExecutor({"a": "parallel", "b": "parallel", "c": "parallel"})
    agent = Agent(
        provider=MultiToolCallProvider(), tools=tools, cwd=tmp_path, tool_execution_mode="parallel"
    )

    await collect(agent, "go")

    assert tools.log.index("start:a") < tools.log.index("end:a")
    assert tools.log[:3] == ["start:a", "start:b", "start:c"]
    assert tools.log[3:] == ["end:a", "end:b", "end:c"]


@pytest.mark.asyncio
async def test_sequential_global_mode_runs_one_at_a_time(tmp_path: Path) -> None:
    tools = RecordingToolExecutor({"a": None, "b": None, "c": None})
    agent = Agent(
        provider=MultiToolCallProvider(),
        tools=tools,
        cwd=tmp_path,
        tool_execution_mode="sequential",
    )

    await collect(agent, "go")

    assert tools.log == ["start:a", "end:a", "start:b", "end:b", "start:c", "end:c"]


@pytest.mark.asyncio
async def test_per_tool_execution_mode_override_breaks_up_parallel_group(
    tmp_path: Path,
) -> None:
    """`b` overrides to "sequential" despite the agent's global "parallel" default, so it
    must run alone, splitting `a` and `c` into their own single-call groups around it."""
    tools = RecordingToolExecutor({"a": "parallel", "b": "sequential", "c": "parallel"})
    agent = Agent(
        provider=MultiToolCallProvider(), tools=tools, cwd=tmp_path, tool_execution_mode="parallel"
    )

    await collect(agent, "go")

    assert tools.log == ["start:a", "end:a", "start:b", "end:b", "start:c", "end:c"]


class SingleToolCallProvider(ModelProvider):
    """Requests one tool call, then finishes on the next turn."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        if messages[-1].role == "toolResult":
            message = AssistantMessage(content=[TextContent(text="done")], stopReason="stop")
            yield AssistantMessageEvent(type="start", partial=message)
            yield AssistantMessageEvent(type="done", reason="stop", message=message)
            return
        call = ToolCall(id="1", name="a", arguments={})
        message = AssistantMessage(content=[call], stopReason="toolUse")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="done", reason="toolUse", message=message)


class RaisingToolExecutor:
    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name="a", description="", parameters={})]

    async def run(self, call: ToolCall, ctx) -> tuple[AgentToolResult, bool]:  # noqa: ANN001
        raise RuntimeError("tool exploded")

    def execution_mode_for(self, name: str) -> str | None:
        return None


class TerminatingToolExecutor:
    def __init__(self, terminate: dict[str, bool | None]) -> None:
        self._terminate = terminate

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name=name, description="", parameters={}) for name in self._terminate]

    async def run(self, call: ToolCall, ctx) -> tuple[AgentToolResult, bool]:  # noqa: ANN001
        return AgentToolResult.text("ok", terminate=self._terminate.get(call.name)), False

    def execution_mode_for(self, name: str) -> str | None:
        return "parallel"


@pytest.mark.asyncio
async def test_terminate_true_on_single_call_ends_run_without_another_turn(
    tmp_path: Path,
) -> None:
    tools = TerminatingToolExecutor({"a": True})
    agent = Agent(provider=SingleToolCallProvider(), tools=tools, cwd=tmp_path)

    events = await collect(agent, "go")

    assert [event.type for event in events].count("turn_start") == 1
    assert events[-1].type == "agent_end"


@pytest.mark.asyncio
async def test_terminate_requires_every_result_in_batch_to_agree(tmp_path: Path) -> None:
    tools = TerminatingToolExecutor({"a": True, "b": None, "c": True})
    agent = Agent(provider=MultiToolCallProvider(), tools=tools, cwd=tmp_path)

    events = await collect(agent, "go")

    assert [event.type for event in events].count("turn_start") == 2
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[-1].content[0].text == "done"


@pytest.mark.asyncio
async def test_tool_executor_exception_becomes_tool_error_and_run_continues(
    tmp_path: Path,
) -> None:
    agent = Agent(provider=SingleToolCallProvider(), tools=RaisingToolExecutor(), cwd=tmp_path)

    events = await collect(agent, "go")

    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert tool_results[0].isError is True
    assert "tool exploded" in tool_results[0].content[0].text
    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[-1].content[0].text == "done"


@pytest.mark.asyncio
async def test_after_tool_call_hook_exception_is_caught_by_outer_net(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content", encoding="utf-8")

    async def after_tool_call(
        call: ToolCall, result: AgentToolResult, is_error: bool
    ) -> ToolResultHookResult | None:
        raise RuntimeError("hook exploded")

    agent = Agent(
        provider=MockProvider(),
        tools=default_registry(tmp_path),
        cwd=tmp_path,
        after_tool_call=after_tool_call,
    )

    events = await collect(agent, "list files")

    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[-1].stopReason == "error"
    assert "hook exploded" in assistant_messages[-1].errorMessage


class ErrorWithToolCallProvider(ModelProvider):
    """Reports a terminal error but (incorrectly/defensively) still includes a tool call."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: object | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        call = ToolCall(id="1", name="a", arguments={})
        message = AssistantMessage(
            content=[call],
            stopReason="error",
            errorMessage="provider failed",
        )
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="error", reason="error", error=message)


class AbortingToolExecutor:
    """Calls `agent.abort()` while executing "a", then records which calls actually ran."""

    def __init__(self) -> None:
        self.agent: Agent | None = None
        self.log: list[str] = []

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name=name, description="", parameters={}) for name in ("a", "b", "c")]

    async def run(self, call: ToolCall, ctx) -> tuple[AgentToolResult, bool]:  # noqa: ANN001
        self.log.append(call.name)
        if call.name == "a":
            assert self.agent is not None
            self.agent.abort()
        return AgentToolResult.text("ok"), False

    def execution_mode_for(self, name: str) -> str | None:
        return None


@pytest.mark.asyncio
async def test_abort_called_early_does_not_hang_the_run(tmp_path: Path) -> None:
    agent = Agent(provider=MockProvider(), cwd=tmp_path)

    gen = agent.run("hello")
    first_event = await gen.__anext__()
    assert first_event.type == "agent_start"
    agent.abort()

    remaining = [event async for event in gen]

    assert remaining[-1].type == "agent_end"


@pytest.mark.asyncio
async def test_abort_skips_not_yet_started_calls_and_ends_run(tmp_path: Path) -> None:
    tools = AbortingToolExecutor()
    agent = Agent(
        provider=MultiToolCallProvider(),
        tools=tools,
        cwd=tmp_path,
        tool_execution_mode="sequential",
    )
    tools.agent = agent

    events = await collect(agent, "go")

    assert tools.log == ["a"]
    tool_results = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult"
    ]
    assert [tr.isError for tr in tool_results] == [False, True, True]
    assert "Operation aborted" in tool_results[1].content[0].text
    assert "Operation aborted" in tool_results[2].content[0].text
    assert [event.type for event in events].count("turn_start") == 1
    assert events[-1].type == "agent_end"


@pytest.mark.asyncio
async def test_provider_error_stop_reason_skips_tool_dispatch(tmp_path: Path) -> None:
    agent = Agent(
        provider=ErrorWithToolCallProvider(), tools=default_registry(tmp_path), cwd=tmp_path
    )

    events = await collect(agent, "go")

    assert "tool_execution_start" not in [event.type for event in events]
    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[-1].stopReason == "error"


@pytest.mark.asyncio
async def test_steer_before_run_reaches_first_provider_call(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path)
    agent.steer("steer-msg")

    await collect(agent, "hi")

    contents = [
        message.content for message in provider.messages if getattr(message, "role", None) == "user"
    ]
    assert contents == ["hi", "steer-msg"]


@pytest.mark.asyncio
async def test_followup_message_after_final_response_prevents_agent_end(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path)
    agent.follow_up("continue")

    turn_starts = 0
    async for event in agent.run("hi"):
        if event.type == "turn_start":
            turn_starts += 1

    assert turn_starts == 2
    contents = [
        message.content for message in provider.messages if getattr(message, "role", None) == "user"
    ]
    assert contents == ["hi", "continue"]


@pytest.mark.asyncio
async def test_steering_takes_priority_over_followup_at_stop_point(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path)
    agent.follow_up("followup-msg")

    turn_starts = 0
    async for event in agent.run("hi"):
        if event.type == "turn_start":
            turn_starts += 1
        if (
            event.type == "message_end"
            and getattr(event.message, "role", None) == "assistant"
            and turn_starts == 1
        ):
            agent.steer("steer-msg")

    assert turn_starts == 3
    contents = [
        message.content for message in provider.messages if getattr(message, "role", None) == "user"
    ]
    assert contents == ["hi", "steer-msg", "followup-msg"]


@pytest.mark.asyncio
async def test_followup_queued_but_iteration_cap_exhausted_still_stops(tmp_path: Path) -> None:
    provider = CapturingProvider()
    agent = Agent(provider=provider, cwd=tmp_path, max_tool_iterations=0)
    agent.follow_up("continue")

    events = await collect(agent, "hi")

    assert events[-1].type == "agent_end"
    assistant_messages = [
        event.message
        for event in events
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant"
    ]
    assert assistant_messages[-1].stopReason == "error"
    assert "Stopped after 0 tool iterations" in assistant_messages[-1].errorMessage
