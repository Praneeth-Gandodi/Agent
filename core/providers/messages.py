from typing import Any, Dict

from anthropic import AsyncAnthropic

from core.providers.base import BaseProvider
from core.providers.models import ChatParams, ChatResponse, ChatStreamingResponse


class AnthropicMessages(BaseProvider):
    def __init__(self, endpoint: str, api_key: str):
        super().__init__(endpoint, api_key)
        self.async_client = AsyncAnthropic(base_url=self.endpoint, api_key=self.api_key)

    def _build_payload(self, params: ChatParams) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": params.model_name,
            "messages": params.messages,
            "max_tokens": params.max_tokens,
        }
        if params.system_prompt:
            payload["system"] = params.system_prompt
        if params.tool_definitions:
            payload["tools"] = params.tool_definitions
            if params.tool_choice:
                payload["tool_choice"] = params.tool_choice

        if params.reasoning_effort:
            payload["thinking"] = params.reasoning_effort
        elif params.temperature is not None:
            payload["temperature"] = params.temperature

        if params.output_config:
            payload["output_config"] = params.output_config

        return payload

    async def streaming_response(self, params: ChatParams) -> ChatStreamingResponse:
        stream = await self.async_client.messages.create(
            **self._build_payload(params), stream=True
        )

        async for chunk in stream:
            yield chunk

    async def response(self, params: ChatParams) -> ChatResponse:
        return await self.async_client.messages.create(
            **self._build_payload(params), stream=False
        )
