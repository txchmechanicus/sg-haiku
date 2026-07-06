from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from upstream.models import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantResponse,
    Message,
)
from upstream.types import ThinkingLevel, ToolSpec


class ModelProvider(ABC):
    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        reasoning: ThinkingLevel | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Yield AssistantMessageEvent objects for one assistant response."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
    ) -> AssistantResponse:
        final: AssistantMessage | None = None
        async for event in self.stream(messages, tools, system_prompt=system_prompt):
            if event.type == "done" and event.message is not None:
                final = event.message
            elif event.type == "error" and event.error is not None:
                final = event.error
        if final is None:
            final = AssistantMessage(
                stopReason="error",
                errorMessage="Provider produced no final message",
            )
        return AssistantResponse(message=final)
