from typing import Any, AsyncIterator, Iterable, Optional, Union

from anthropic.types import (
    Message,
    MessageParam,
    MessageStreamEvent,
    ThinkingConfigParam,
    ToolChoiceParam,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel


class ChatParams(BaseModel):
    model_name: str
    messages: Optional[
        Union[
            Iterable[ChatCompletionMessageParam],
            Iterable[MessageParam],
        ]
    ] = None
    input: ResponseInputParam | None = None  # For Response API
    system_prompt: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    tool_choice: Optional[
        Union[ChatCompletionToolChoiceOptionParam, ToolChoiceParam]
    ] = None
    temperature: Optional[float] | None = None
    reasoning_effort: Optional[
        Union[ReasoningEffort, ThinkingConfigParam, dict[str, str]]
    ] = None
    max_tokens: int | None = None
    max_tool_calls: int | None = None
    parallel_tool_calls: bool = True
    store: bool = False


OpenAIResponseStreamEvent = Union[
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseCompletedEvent,
]

ChatResponse = Union[ChatCompletion, Response, Message]

ChatStreamingResponse = AsyncIterator[
    Union[ChatCompletionChunk, ResponseStreamEvent, MessageStreamEvent]
]
