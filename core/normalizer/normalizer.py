from core.normalizer.models import (
    NormalizedEvent,
    ReasoningChunk,
    StreamingTextChunk,
    ToolCall,
    ToolCallList,
)
from openai.types.chat import ChatCompletionChunk
from functools import singledispatchmethod
from typing import Any


class ResponseNormalizer:
    def __init__(self) -> None:
        self._tool_calls: dict[int, ToolCall] = {}

    @singledispatchmethod
    def normalize(self, chunk: Any) -> NormalizedEvent | None:
        raise TypeError(f"No normalizer {type(chunk)}")

    @normalize.register
    def _(self, chunk: ChatCompletionChunk) -> NormalizedEvent | None:
        if not chunk.choices or not chunk.choices[0]:
            return None

        if chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if delta.content:
                return StreamingTextChunk(content=delta.content)

            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                return ReasoningChunk(reasoning=reasoning)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index
                    if index not in self._tool_calls:
                        self._tool_calls[index] = ToolCall(index=index)

                    partial_tool_call = self._tool_calls[index]
                    if tc.id:
                        partial_tool_call.id = tc.id
                    if tc.function:
                        if tc.function.name:
                            partial_tool_call.name += tc.function.name
                        if tc.function.arguments:
                            partial_tool_call.arguments += tc.function.arguments

        if chunk.choices[0].finish_reason == "tool_calls" and self._tool_calls:
            completed_tool_calls: ToolCallList = ToolCallList(
                tool_calls=list(self._tool_calls.values())
            )
            self._tool_calls.clear()
            return completed_tool_calls
