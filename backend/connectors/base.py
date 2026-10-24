"""Base connector protocol and domain errors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.connectors.schemas import ConnectorHealth, ConnectorSearchBatch, ExternalResource, ResourceQuery
from backend.core.errors import CourseAgentError


class ConnectorError(CourseAgentError):
    error_code = "CONNECTOR_ERROR"


class ConnectorTimeoutError(ConnectorError):
    error_code = "CONNECTOR_TIMEOUT"


class ConnectorRateLimitError(ConnectorError):
    error_code = "CONNECTOR_RATE_LIMIT"


class ConnectorBadRequestError(ConnectorError):
    error_code = "CONNECTOR_UPSTREAM_BAD_REQUEST"


class ConnectorAuthenticationError(ConnectorError):
    error_code = "CONNECTOR_UPSTREAM_AUTHENTICATION_FAILED"


class ConnectorForbiddenError(ConnectorError):
    error_code = "CONNECTOR_UPSTREAM_FORBIDDEN"


class ConnectorNotFoundError(ConnectorError):
    error_code = "CONNECTOR_UPSTREAM_NOT_FOUND"


class ConnectorServiceError(ConnectorError):
    error_code = "CONNECTOR_SERVICE_ERROR"


class ConnectorConfigurationError(ConnectorError):
    error_code = "CONNECTOR_CONFIGURATION_ERROR"


class ExternalConnector(ABC):
    name: str
    enabled: bool = True
    supports_remote: bool = True

    def will_attempt_remote(self, query: ResourceQuery) -> bool:
        """Return whether this validated query can reach the remote transport."""
        return self.supports_remote

    @abstractmethod
    async def health_check(self) -> ConnectorHealth:
        ...

    @abstractmethod
    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        ...

    async def search_with_metadata(self, query: ResourceQuery) -> ConnectorSearchBatch:
        return ConnectorSearchBatch(resources=await self.search(query))

    async def get_by_id(self, external_id: str) -> ExternalResource | None:
        query = ResourceQuery(query=external_id, page_size=1)
        results = await self.search(query)
        for item in results:
            if item.external_id == external_id:
                return item
        return None

