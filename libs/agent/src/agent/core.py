from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal, Protocol

from upstream.models import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from upstream.providers import ModelProvider
from upstream.types import AgentToolResult, ToolSpec

from agent.events import AgentEvent

SYSTEM_PROMPT = "You are Haiku, a concise coding agent. Use tools when they help answer the user."


class ToolExecutor(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    async def run(self, call: ToolCall) -> tuple[AgentToolResult, bool]: ...


class EmptyToolExecutor:
    def specs(self) -> list[ToolSpec]:
        return []

    async def run(self, call: ToolCall) -> tuple[AgentToolResult, bool]:
        return AgentToolResult.text(f"Unknown tool: {call.name}"), True


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
    ) -> None:
        self.provider = provider
        self.cwd = (cwd or Path.cwd()).resolve()
        self.tools = tools or EmptyToolExecutor()
        self.max_tool_iterations = max_tool_iterations
        self.system_prompt = system_prompt
        self.tool_execution_mode = tool_execution_mode

    async def _execute_tool(self, call: ToolCall) -> tuple[ToolCall, AgentToolResult, bool]:
        result, is_error = await self.tools.run(call)
        return call, result, is_error

    async def run(
        self,
        prompt: str,
        *,
        initial_messages: list[Message] | None = None,
        system_prompt: str | None = None,
        use_tools: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        messages: list[Message] = list(initial_messages or [])
        user_message = UserMessage(content=prompt)
        messages.append(user_message)
        specs = self.tools.specs() if use_tools else []

        yield AgentEvent.agent_start()
        yield AgentEvent.message_start(user_message)
        yield AgentEvent.message_end(user_message)

        for _ in range(self.max_tool_iterations + 1):
            yield AgentEvent.turn_start()
            assistant_message: AssistantMessage | None = None
            async for assistant_event in self.provider.stream(
                messages,
                specs,
                system_prompt=system_prompt or self.system_prompt,
            ):
                current_message = (
                    assistant_event.partial or assistant_event.message or assistant_event.error
                )
                if current_message is not None:
                    assistant_message = current_message
                    if assistant_event.type == "start":
                        yield AgentEvent.message_start(current_message)
                    yield AgentEvent.message_update(current_message, assistant_event)

            if assistant_message is None:
                assistant_message = AssistantMessage(
                    content=[TextContent(text="Provider produced no assistant message.")],
                    stopReason="error",
                    errorMessage="Provider produced no assistant message.",
                )

            messages.append(assistant_message)
            yield AgentEvent.message_end(assistant_message)

            tool_calls = [
                part for part in assistant_message.content if isinstance(part, ToolCall)
            ]
            if not tool_calls:
                yield AgentEvent.turn_end(assistant_message, [])
                yield AgentEvent.agent_end(messages)
                return

            if not use_tools:
                error_message = ToolResultMessage(
                    toolCallId=tool_calls[0].id,
                    toolName=tool_calls[0].name,
                    content=[TextContent(text="Model requested tools, but tools are disabled.")],
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

            if self.tool_execution_mode == "parallel":
                executed = list(
                    await asyncio.gather(*[self._execute_tool(call) for call in tool_calls])
                )
            else:
                executed = [await self._execute_tool(call) for call in tool_calls]

            tool_results: list[ToolResultMessage] = []
            for call, result, is_error in executed:
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
