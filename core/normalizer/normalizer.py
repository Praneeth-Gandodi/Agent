from core.normalizer.models import (
    NormalizedEvent,
    ReasoningChunk,
    ResponseComplete,
    StreamingTextChunk,
    ToolCall,
    ToolCallList,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion
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

    @normalize.register
    def _(self, chunk: ChatCompletion) -> NormalizedEvent | None:
        result = ResponseComplete()
        if chunk.usage:
            if chunk.usage.completion_tokens:
                result.completion_tokens = chunk.usage.completion_tokens
            if chunk.usage.prompt_tokens:
                result.prompt_tokens = chunk.usage.prompt_tokens
            if chunk.usage.total_tokens:
                result.total_tokens = chunk.usage.total_tokens

        delta = chunk.choices[0]
        if delta.finish_reason:
            result.finish_reason = delta.finish_reason

        if delta.message.content:
            result.content = delta.message.content

        reasoning = getattr(delta.message, "reasoning", None)
        if reasoning:
            result.reasoning_content = reasoning

        if delta.message.tool_calls:
            tool_calls = []

            for tc in delta.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id or "",
                        name=tc.function.name if tc.function else "",  # type: ignore
                        arguments=tc.function.arguments if tc.function else "",  # type: ignore
                    )
                )

            result.tool_calls = ToolCallList(tool_calls=tool_calls)

        return result
