from pydantic import BaseModel
from typing import Union


class StreamingTextChunk(BaseModel):
    content: str | None = None
    finish_reason: str | None = None


class ReasoningChunk(BaseModel):
    reasoning: str


class ResponseComplete(BaseModel):
    content: str
    reasoning_content: str | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None


class ToolCall(BaseModel):
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


class ToolCallList(BaseModel):
    tool_calls: list[ToolCall] = []


NormalizedEvent = Union[
    StreamingTextChunk, ResponseComplete, ToolCallList, ReasoningChunk
]
