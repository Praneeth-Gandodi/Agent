from typing import Union

from pydantic import BaseModel


class StreamingTextChunk(BaseModel):
    content: str | None = None
    finish_reason: str | None = None


class StreamingTextComplete(BaseModel):
    content: str


class ReasoningChunk(BaseModel):
    reasoning: str | None = None
    reasoning_summary: str | None = None


class ToolCall(BaseModel):
    index: int | str | None = None
    id: str | None = None
    call_id: str | None = None
    name: str = ""
    arguments: str = ""


class FinishReason(BaseModel):
    finish_reason: str


class ToolCallList(BaseModel):
    tool_calls: list[ToolCall] = []


class TokenUsageDetails(BaseModel):
    completion_tokens: int | None = None
    input_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


class NormalizerResponseComplete(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    reasoning_summary: list | str | None = None
    tool_calls: ToolCallList | None = None
    finish_reason: str | None = None
    token_details: TokenUsageDetails | None = None


NormalizedEvent = Union[
    StreamingTextChunk,
    StreamingTextComplete,
    NormalizerResponseComplete,
    ToolCallList,
    ToolCall,
    ReasoningChunk,
    FinishReason,
]
