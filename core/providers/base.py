from abc import ABC, abstractmethod
from typing import Any, Dict

from core.providers.models import ChatParams, ChatResponse, ChatStreamingResponse


class BaseProvider(ABC):
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key

    @abstractmethod
    def _build_payload(self, params: ChatParams) -> Dict[str, Any]:
        pass

    @abstractmethod
    def streaming_response(self, params: ChatParams) -> ChatStreamingResponse:
        pass

    @abstractmethod
    async def response(self, params: ChatParams) -> ChatResponse:
        pass
