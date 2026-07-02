from __future__ import annotations

from agent import AgentEvent
from upstream import (
    AssistantMessage,
    AssistantMessageEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def test_user_message_serialization_matches_contract_shape() -> None:
    message = UserMessage(content="hello", timestamp=123)

    assert message.model_dump(mode="json", exclude_none=True) == {
        "role": "user",
        "content": "hello",
        "timestamp": 123,
    }


def test_assistant_message_serialization_matches_contract_shape() -> None:
    call = ToolCall(id="call-1", name="ls", arguments={"path": "."})
    message = AssistantMessage(content=[call], stopReason="toolUse", timestamp=123)

    data = message.model_dump(mode="json", exclude_none=True)

    assert data["role"] == "assistant"
    assert data["content"][0]["type"] == "toolCall"
    assert data["content"][0]["id"] == "call-1"
    assert data["stopReason"] == "toolUse"


def test_tool_result_message_serialization_matches_contract_shape() -> None:
    message = ToolResultMessage(
        toolCallId="call-1",
        toolName="ls",
        content=[TextContent(text="ok")],
        isError=False,
        timestamp=123,
    )

    data = message.model_dump(mode="json", exclude_none=True)

    assert data == {
        "role": "toolResult",
        "toolCallId": "call-1",
        "toolName": "ls",
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
        "timestamp": 123,
    }


def test_agent_event_serialization_matches_contract_shape() -> None:
    message = AssistantMessage(content=[TextContent(text="hello")], timestamp=123)
    assistant_event = AssistantMessageEvent(
        type="text_delta",
        contentIndex=0,
        delta="hello",
        partial=message,
    )
    event = AgentEvent.message_update(message, assistant_event)

    data = event.model_dump(mode="json", exclude_none=True)

    assert data["type"] == "message_update"
    assert data["message"]["role"] == "assistant"
    assert data["assistantMessageEvent"]["type"] == "text_delta"
    assert data["assistantMessageEvent"]["delta"] == "hello"
