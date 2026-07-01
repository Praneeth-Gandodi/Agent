from functools import singledispatchmethod
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from anthropic.types import Message, RawMessageStreamEvent

from core.normalizer.models import (
    FinishReason,
    NormalizedEvent,
    ReasoningChunk,
    NormalizerResponseComplete,
    StreamingTextChunk,
    TokenUsageDetails,
    ToolCall,
    ToolCallList,
)
from core.providers import OpenAIResponseStreamEvent


class ResponseNormalizer:
    def __init__(self) -> None:
        self._tool_calls: dict[int | str, ToolCall] = {}

    @singledispatchmethod
    def normalize(self, chunk: Any) -> NormalizedEvent | None:
        raise TypeError(f"No normalizer for the type {type(chunk)}")

    @normalize.register
    def _(self, chunk: ChatCompletionChunk) -> NormalizedEvent | None:
        if not chunk.choices or not chunk.choices[0]:
            if chunk.usage:
                result = NormalizerResponseComplete()
                result.token_details = TokenUsageDetails(
                    completion_tokens=chunk.usage.completion_tokens,
                    prompt_tokens=chunk.usage.prompt_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                if (
                    chunk.usage.completion_tokens_details
                    and chunk.usage.completion_tokens_details.reasoning_tokens
                ):
                    result.token_details.reasoning_tokens = (
                        chunk.usage.completion_tokens_details.reasoning_tokens
                    )
                return result
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

        if chunk.usage:
            result = NormalizerResponseComplete()
            result.token_details = TokenUsageDetails(
                completion_tokens=chunk.usage.completion_tokens,
                prompt_tokens=chunk.usage.prompt_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
            if (
                chunk.usage.completion_tokens_details
                and chunk.usage.completion_tokens_details.reasoning_tokens
            ):
                result.token_details.reasoning_tokens = (
                    chunk.usage.completion_tokens_details.reasoning_tokens
                )
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason:
                result.finish_reason = finish_reason
            return result

        if chunk.choices[0].finish_reason:
            return FinishReason(finish_reason=chunk.choices[0].finish_reason)

    @normalize.register
    def _(self, chunk: ChatCompletion) -> NormalizedEvent | None:
        result = NormalizerResponseComplete()
        if chunk.usage:
            result.token_details = TokenUsageDetails(
                completion_tokens=chunk.usage.completion_tokens,
                prompt_tokens=chunk.usage.prompt_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
            if (
                chunk.usage.completion_tokens_details
                and chunk.usage.completion_tokens_details.reasoning_tokens
            ):
                result.token_details.reasoning_tokens = (
                    chunk.usage.completion_tokens_details.reasoning_tokens
                )

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

    @normalize.register
    def _(self, chunk: OpenAIResponseStreamEvent) -> NormalizedEvent | None:
        match chunk.type:
            case "response.output_text.delta":
                return StreamingTextChunk(content=chunk.delta)

            case "response.reasoning_text.delta":
                return ReasoningChunk(reasoning=chunk.delta)

            case "response.reasoning_summary_text.delta":
                return ReasoningChunk(reasoning=chunk.delta)

            case "response.output_item.added":
                item = chunk.item

                if item.type == "function_call":
                    if item.id:
                        self._tool_calls[item.id] = ToolCall(
                            id=item.id,
                            call_id=item.call_id,
                            name=item.name,
                            arguments=item.arguments if item.arguments else "",
                        )

            case "response.function_call_arguments.delta":
                if (
                    chunk.item_id in self._tool_calls
                    and self._tool_calls[chunk.item_id]
                ):
                    self._tool_calls[chunk.item_id].arguments += chunk.delta

            case "response.output_item.done":
                if chunk.item.type == "function_call" and chunk.item.id in self._tool_calls:
                    return ToolCallList(tool_calls=[self._tool_calls.pop(chunk.item.id)])
            case "response.completed":
                result = NormalizerResponseComplete()
                if chunk.response.output_text:
                    result.content = chunk.response.output_text
                for event in chunk.response.output:
                    if event.type == "reasoning":
                        result.reasoning_summary = event.summary
                        if event.content:
                            result.reasoning_content = event.content[0].text

                    if event.type == "function_call":
                        if result.tool_calls is None:
                            result.tool_calls = ToolCallList()
                        result.tool_calls.tool_calls.append(
                            ToolCall(
                                id=event.id if event.id else "",
                                call_id=event.call_id,
                                name=event.name,
                                arguments=event.arguments,
                            )
                        )

                if chunk.response.usage:
                    result.token_details = TokenUsageDetails()
                    if chunk.response.usage.total_tokens:
                        result.token_details.total_tokens = (
                            chunk.response.usage.total_tokens
                        )
                    if chunk.response.usage.output_tokens_details:
                        result.token_details.reasoning_tokens = (
                            chunk.response.usage.output_tokens_details.reasoning_tokens
                        )
                    if chunk.response.usage.input_tokens:
                        result.token_details.input_tokens = (
                            chunk.response.usage.input_tokens
                        )

                return result

    @normalize.register
    def _(self, chunk: Response) -> NormalizedEvent | None:
        result = NormalizerResponseComplete()
        if chunk.output_text:
            result.content = chunk.output_text
        for event in chunk.output:
            if event.type == "reasoning":
                result.reasoning_summary = event.summary
                if event.content:
                    result.reasoning_content = event.content[0].text

            if event.type == "function_call":
                if result.tool_calls is None:
                    result.tool_calls = ToolCallList()
                result.tool_calls.tool_calls.append(
                    ToolCall(
                        id=event.id if event.id else "",
                        call_id=event.call_id,
                        name=event.name,
                        arguments=event.arguments,
                    )
                )

        if chunk.usage:
            result.token_details = TokenUsageDetails()
            if chunk.usage.total_tokens:
                result.token_details.total_tokens = chunk.usage.total_tokens
            if chunk.usage.output_tokens_details:
                result.token_details.reasoning_tokens = (
                    chunk.usage.output_tokens_details.reasoning_tokens
                )
            if chunk.usage.input_tokens:
                result.token_details.input_tokens = chunk.usage.input_tokens

        return result

    @normalize.register
    def _(self, chunk: RawMessageStreamEvent) -> NormalizedEvent | None:
        match chunk.type:
            case "content_block_start":
                if chunk.content_block.type == "tool_use":
                    self._tool_calls[chunk.index] = ToolCall(
                        id=chunk.content_block.id,
                        name=chunk.content_block.name,
                        arguments="",
                    )
                return None

            case "content_block_delta":
                if chunk.delta.type == "thinking_delta":
                    return ReasoningChunk(reasoning=chunk.delta.thinking)
                if chunk.delta.type == "text_delta":
                    return StreamingTextChunk(content=chunk.delta.text)
                if chunk.delta.type == "input_json_delta":
                    if chunk.index in self._tool_calls:
                        self._tool_calls[
                            chunk.index
                        ].arguments += chunk.delta.partial_json
                return None

            case "content_block_stop":
                if chunk.index in self._tool_calls:
                    return self._tool_calls.pop(chunk.index)
                return None

            case "message_delta":
                result = NormalizerResponseComplete()
                if chunk.delta.stop_reason:
                    result.finish_reason = chunk.delta.stop_reason
                if chunk.usage:
                    result.token_details = TokenUsageDetails(
                        completion_tokens=chunk.usage.output_tokens,
                    )

                return result

            case "message_start" | "message_stop":
                return None

    @normalize.register
    def _(self, chunk: Message) -> NormalizedEvent | None:
        result = NormalizerResponseComplete()

        if chunk.usage:
            result.token_details = TokenUsageDetails()
            result.token_details.input_tokens = chunk.usage.input_tokens
            result.token_details.completion_tokens = chunk.usage.output_tokens

        if chunk.stop_reason:
            result.finish_reason = chunk.stop_reason

        for event in chunk.content:
            if event.type == "thinking":
                result.reasoning_content = event.thinking

            if event.type == "text":
                result.content = event.text

            if event.type == "tool_use":
                self._tool_calls[event.id] = ToolCall(
                    id=event.id,
                    name=event.name,
                    arguments=str(event.input) if event.input else "",
                )

        if self._tool_calls:
            result.tool_calls = ToolCallList(tool_calls=list(self._tool_calls.values()))
            self._tool_calls.clear()

        return result
