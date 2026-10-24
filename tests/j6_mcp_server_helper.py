from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from backend.connectors.base import ConnectorServiceError, ExternalConnector
from backend.connectors.registry import ConnectorRegistry
from backend.connectors.schemas import ConnectorHealth, ExternalResource, ResourceQuery
from mcp_server import tools as tool_module
from mcp_server.server import main


class ControlledArxivConnector(ExternalConnector):
    name = "arxiv"
    enabled = True
    supports_remote = True

    async def health_check(self) -> ConnectorHealth:
        return ConnectorHealth(name=self.name, enabled=True, status="ok")

    async def search(self, query: ResourceQuery) -> list[ExternalResource]:
        if os.getenv("J6_CONNECTOR_MODE", "fail") == "fail":
            raise ConnectorServiceError("controlled upstream failure", error_code="CONNECTOR_SERVICE_ERROR")
        delay = float(os.getenv("J6_DELAY_SECONDS", "0.5"))
        marker = Path(os.environ["J6_MARKER_PATH"])
        started = time.time()
        marker.write_text(json.dumps({"started_at": started}), encoding="utf-8")
        time.sleep(delay)
        ended = time.time()
        marker.write_text(json.dumps({"started_at": started, "ended_at": ended}), encoding="utf-8")
        return [
            ExternalResource(
                external_id="j6-slow-paper",
                resource_type="paper",
                source_name="arxiv",
                title="J6 delayed paper",
                abstract="Controlled delayed connector result.",
                url="https://arxiv.org/abs/j6-slow-paper",
            )
        ]


tool_module.get_default_registry = lambda: ConnectorRegistry([ControlledArxivConnector()])
tool_module.TOOL_DEFINITIONS["paper_search_real"] = tool_module.TOOL_DEFINITIONS[
    "paper_search_real"
].model_copy(
    update={
        "enabled": True,
        "timeout_seconds": float(os.getenv("J6_TOOL_TIMEOUT_SECONDS", "0.15")),
    }
)


if __name__ == "__main__":
    asyncio.run(main())
