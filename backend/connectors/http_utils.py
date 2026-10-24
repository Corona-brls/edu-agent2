"""Shared HTTP helpers for external connectors."""

from __future__ import annotations

import httpx

from backend.connectors.base import (
    ConnectorAuthenticationError,
    ConnectorBadRequestError,
    ConnectorForbiddenError,
    ConnectorNotFoundError,
    ConnectorRateLimitError,
    ConnectorServiceError,
)


def map_http_error(response: httpx.Response) -> None:
    status = response.status_code
    if status == 400:
        raise ConnectorBadRequestError("外部服务拒绝了查询参数。")
    if status == 401:
        raise ConnectorAuthenticationError("外部服务凭证无效。")
    if status == 403:
        raise ConnectorForbiddenError("外部服务拒绝访问。")
    if status == 404:
        raise ConnectorNotFoundError("外部服务未找到指定资源。")
    if status == 429:
        raise ConnectorRateLimitError("外部服务限流。", error_code="CONNECTOR_RATE_LIMIT")
    if status >= 500:
        raise ConnectorServiceError("外部服务暂时不可用。", error_code="CONNECTOR_SERVICE_ERROR")
    response.raise_for_status()

