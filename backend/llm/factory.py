"""Factory for selecting the configured LLM provider."""

from __future__ import annotations

from typing import Any

from backend.core.config import get_settings
from backend.llm.base import LLMProvider
from backend.llm.deepseek_provider import DeepSeekProvider
from backend.llm.template_provider import TemplateProvider


def create_llm_provider(client: Any | None = None) -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "deepseek" and settings.deepseek_api_key:
        try:
            return DeepSeekProvider(settings=settings, client=client)
        except Exception:
            if settings.llm_fallback_provider == "template":
                return TemplateProvider()
            raise
    return TemplateProvider()
