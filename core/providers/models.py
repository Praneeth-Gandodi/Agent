from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.responses import ResponseStreamEvent
from anthropic.types import (
    Message,
    MessageParam,
    MessageStreamEvent,
    ThinkingConfigParam,
    ToolUnionParam,
    ToolChoiceParam,
)


from typing import Iterable, Union, Optional, AsyncIterator


class ChatParams(BaseModel):
    model_name: str
    messages: Union[Iterable[ChatCompletionMessageParam], Iterable[MessageParam]]
    system_prompt: str | None = None
    tool_definitions: Optional[
        Union[Iterable[ChatCompletionToolUnionParam], Iterable[ToolUnionParam]],
    ] = None
    tool_choice: Optional[
        Union[ChatCompletionToolChoiceOptionParam, ToolChoiceParam]
    ] = None
    temperature: Optional[float] | None = None
    reasoning_effort: Optional[Union[ReasoningEffort, ThinkingConfigParam]] = None
    max_tokens: int | None = None
    parallel_tool_calls: bool = True


ChatResponse = Union[ChatCompletion, Message]
ChatStreamingResponse = AsyncIterator[Union[ChatCompletionChunk, MessageStreamEvent]]
