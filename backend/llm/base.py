"""Abstract interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from backend.agent.schemas import ActionDecision, ToolSpec
from backend.llm.schemas import LLMRequest, LLMResponse

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class LLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate_answer(self, request: LLMRequest) -> LLMResponse:
        """Generate natural language answer from evidence."""

    @abstractmethod
    def generate_structured(
        self,
        request: LLMRequest,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        """Generate and validate structured output."""

    @abstractmethod
    def decide_next_action(
        self,
        state_summary: str,
        available_tools: list[ToolSpec],
    ) -> ActionDecision:
        """Return the next agent action without executing it."""
