from core.providers.models import ChatParams, ChatResponse, ChatStreamingResponse
from core.providers.base import BaseProvider
from typing import Dict, Any
from openai import AsyncOpenAI


class OpenAIChatCompletion(BaseProvider):
    def __init__(self, endpoint: str, api_key: str):
        super().__init__(endpoint, api_key)
        self.async_client = AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key)

    def _build_payload(self, params: ChatParams) -> Dict[str, Any]:
        messages: list[dict[str, str]] = []
        if params.system_prompt:
            messages = [{"role": "system", "content": params.system_prompt}]
            messages.extend(list(params.messages))  # pyright: ignore
        payload: Dict[str, Any] = {
            "model": params.model_name,
            "messages": params.messages if not messages else messages,
            "max_tokens": params.max_tokens,
        }
        if params.tool_definitions:
            payload["tools"] = params.tool_definitions
            payload["parallel_tool_calls"] = params.parallel_tool_calls
            if params.tool_choice:
                payload["tool_choice"] = params.tool_choice
        if params.reasoning_effort:
            payload["reasoning_effort"] = params.reasoning_effort
        elif params.temperature is not None:
            payload["temperature"] = params.temperature

        return payload

    async def streaming_response(self, params: ChatParams) -> ChatStreamingResponse:
        stream = await self.async_client.chat.completions.create(
            **self._build_payload(params), stream=True
        )
        async for chunk in stream:
            yield chunk

    async def response(self, params: ChatParams) -> ChatResponse:
        return await self.async_client.chat.completions.create(
            **self._build_payload(params), stream=False
        )
