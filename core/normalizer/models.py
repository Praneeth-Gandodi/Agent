from pydantic import BaseModel
from typing import Union


class StreamingTextChunk(BaseModel):
    content: str | None = None
    finish_reason: str | None = None


class ReasoningChunk(BaseModel):
    reasoning: str


class ToolCall(BaseModel):
    index: int | None = None
    id: str = ""
    name: str = ""
    arguments: str = ""


class ToolCallList(BaseModel):
    tool_calls: list[ToolCall] = []


class ResponseComplete(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: ToolCallList | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None


NormalizedEvent = Union[
    StreamingTextChunk, ResponseComplete, ToolCallList, ReasoningChunk
]
