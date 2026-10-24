"""Local read-only tools exposed through the MCP server."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.connectors.nvd import nvd_query_metadata
from backend.connectors.registry import get_default_registry
from backend.connectors.schemas import ResourceQuery
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.retriever import LocalRetriever
from backend.resources.service import ExternalResourceService
from backend.storage.logger import LearningLogger
from mcp_server.schemas import (
    CourseRetrieverInput,
    GitHubProjectSearchInput,
    MermaidGeneratorInput,
    PaperSearchInput,
    RealPaperSearchInput,
    RealStandardLookupInput,
    RealVulnLookupInput,
    StandardLookupInput,
    StandardMetadataLookupInput,
    ToolDefinition,
    ToolResult,
    VulnLookupInput,
)


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "course_retriever": ToolDefinition(
        name="course_retriever",
        description="检索本地课程知识库。",
        agent_description=(
            "何时使用：用户询问课程内容、专题概念、实验设计证据时使用。"
            "何时不应使用：用户要求真实联网论文、实时漏洞库或执行代码时不使用。"
        ),
        input_schema=CourseRetrieverInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=5,
        network=False,
        read_only=True,
        enabled=True,
    ),
    "paper_search": ToolDefinition(
        name="paper_search",
        description="检索本地论文样例和研究方向资料。",
        agent_description=(
            "何时使用：用户询问论文样例、研究方向、文献综述线索时使用。"
            "何时不应使用：用户要求真实联网论文数据库或具体未上传论文结论时不使用。"
        ),
        input_schema=PaperSearchInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=5,
        network=False,
        read_only=True,
        enabled=True,
    ),
    "vuln_lookup": ToolDefinition(
        name="vuln_lookup",
        description="检索本地漏洞和 CVE 样例。",
        agent_description=(
            "何时使用：用户询问漏洞样例、CVE、PoC 或课程实验漏洞素材时使用。"
            "何时不应使用：用户要求实时 NVD 查询或编造不存在 CVE 时不使用。"
        ),
        input_schema=VulnLookupInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=5,
        network=False,
        read_only=True,
        enabled=True,
    ),
    "standard_lookup": ToolDefinition(
        name="standard_lookup",
        description="检索本地标准文档样例片段。",
        agent_description=(
            "何时使用：用户询问 3GPP、TS 24.501 或标准相关课程片段时使用。"
            "何时不应使用：用户要求真实标准全文或精确未收录条款时不使用。"
        ),
        input_schema=StandardLookupInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=5,
        network=False,
        read_only=True,
        enabled=True,
    ),
    "mermaid_generator": ToolDefinition(
        name="mermaid_generator",
        description="生成 Mermaid 架构图或流程图文本。",
        agent_description=(
            "何时使用：用户明确要求生成系统架构、流程图或 Mermaid 图时使用。"
            "何时不应使用：用户只需要事实问答或证据检索时不使用。"
        ),
        input_schema=MermaidGeneratorInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=3,
        network=False,
        read_only=True,
        enabled=True,
    ),
    "paper_search_real": ToolDefinition(
        name="paper_search_real",
        description="Query enabled paper metadata connectors.",
        agent_description=(
            "何时使用：本地证据不足且需要真实论文标题、作者、摘要、DOI/arXiv ID 和来源链接时使用。"
            "何时不应使用：本地资料足够、联网工具关闭、需要论文全文或不能验证来源时不使用。"
        ),
        input_schema=RealPaperSearchInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=25,
        network=True,
        read_only=True,
        enabled=os.getenv("ENABLE_REAL_PAPER_SEARCH", "false").strip().lower() in {"1", "true", "yes", "on"},
    ),
    "vuln_lookup_real": ToolDefinition(
        name="vuln_lookup_real",
        description="Query enabled NVD CVE metadata connector.",
        agent_description=(
            "何时使用：本地漏洞样例不足且需要真实 CVE、CVSS、CWE、发布日期、更新时间和来源链接时使用。"
            "何时不应使用：用户要求利用代码、攻击步骤或联网工具关闭时不使用。"
        ),
        input_schema=RealVulnLookupInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=25,
        network=True,
        read_only=True,
        enabled=os.getenv("ENABLE_REAL_CVE_LOOKUP", "false").strip().lower() in {"1", "true", "yes", "on"},
    ),
    "standard_metadata_lookup": ToolDefinition(
        name="standard_metadata_lookup",
        description="Query standard metadata and local upload status; no copyrighted full text fetching.",
        agent_description=(
            "何时使用：需要标准编号、版本、官方链接或本地是否已上传正文时使用。"
            "何时不应使用：用户要求自动抓取标准全文或绕过版权限制时不使用。"
        ),
        input_schema=StandardMetadataLookupInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=5,
        network=False,
        read_only=True,
        enabled=True,
    ),
    "standard_lookup_real": ToolDefinition(
        name="standard_lookup_real",
        description="Query RFC Editor metadata when enabled, with verified local standard fallback.",
        agent_description=(
            "何时使用：需要精确 RFC 元数据且 RFC Editor 来源已启用时使用。"
            "何时不应使用：3GPP 本地目录查询、联网来源关闭或要求标准全文时不使用。"
        ),
        input_schema=RealStandardLookupInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=25,
        network=True,
        read_only=True,
        enabled=os.getenv("ENABLE_REAL_RFC_LOOKUP", "false").strip().lower() in {"1", "true", "yes", "on"},
    ),
    "github_project_search": ToolDefinition(
        name="github_project_search",
        description="Optionally query GitHub public repository metadata; never executes code.",
        agent_description=(
            "何时使用：需要开源项目案例推荐且 GitHub Connector 已启用时使用。"
            "何时不应使用：用户要求执行仓库代码、下载不可信脚本或工具关闭时不使用。"
        ),
        input_schema=GitHubProjectSearchInput.model_json_schema(),
        risk="LOW",
        timeout_seconds=25,
        network=True,
        read_only=True,
        enabled=os.getenv("ENABLE_GITHUB_SEARCH", "false").strip().lower() in {"1", "true", "yes", "on"},
    ),
}


_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "course_retriever": CourseRetrieverInput,
    "paper_search": PaperSearchInput,
    "vuln_lookup": VulnLookupInput,
    "standard_lookup": StandardLookupInput,
    "mermaid_generator": MermaidGeneratorInput,
    "paper_search_real": RealPaperSearchInput,
    "vuln_lookup_real": RealVulnLookupInput,
    "standard_metadata_lookup": StandardMetadataLookupInput,
    "standard_lookup_real": RealStandardLookupInput,
    "github_project_search": GitHubProjectSearchInput,
}

_INPUT_MODELS_VIEW = MappingProxyType(_INPUT_MODELS)


def get_tool_input_models() -> Mapping[str, type[BaseModel]]:
    """Return the canonical, read-only input-model registry for MCP tools."""

    return _INPUT_MODELS_VIEW


def get_tool_input_model(name: str) -> type[BaseModel] | None:
    return _INPUT_MODELS.get(name)


def list_tool_definitions() -> list[ToolDefinition]:
    return list(TOOL_DEFINITIONS.values())


def call_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    definition = TOOL_DEFINITIONS.get(name)
    if definition is None:
        return _error("TOOL_NOT_FOUND", f"工具不存在：{name}", {"tool_name": name})
    if not definition.enabled:
        return _error("TOOL_DISABLED", f"工具已禁用：{name}", {"tool_name": name})

    input_model = _INPUT_MODELS[name]
    try:
        payload = input_model.model_validate(arguments)
    except ValidationError as exc:
        return _error(
            "INVALID_ARGUMENTS",
            "工具参数不符合 schema",
            {"tool_name": name, "errors": exc.errors()},
        )

    handlers: dict[str, Callable[[BaseModel], dict[str, Any]]] = {
        "course_retriever": _course_retriever,
        "paper_search": _paper_search,
        "vuln_lookup": _vuln_lookup,
        "standard_lookup": _standard_lookup,
        "mermaid_generator": _mermaid_generator,
        "paper_search_real": _paper_search_real,
        "vuln_lookup_real": _vuln_lookup_real,
        "standard_metadata_lookup": _standard_metadata_lookup,
        "standard_lookup_real": _standard_lookup_real,
        "github_project_search": _github_project_search,
    }
    def invoke(deadline: float) -> dict[str, Any]:
        if name in {
            "paper_search_real",
            "vuln_lookup_real",
            "standard_metadata_lookup",
            "standard_lookup_real",
            "github_project_search",
        }:
            return handlers[name](payload, deadline=deadline)
        return handlers[name](payload)

    return _run_with_timeout(
        invoke,
        timeout_seconds=definition.timeout_seconds,
        tool_name=name,
    )


def _run_with_timeout(
    fn: Callable[[float], dict[str, Any]],
    timeout_seconds: float,
    tool_name: str,
) -> ToolResult:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_seconds
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn, deadline)
        data = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        return _error(
            "TOOL_TIMEOUT",
            f"工具调用超时：{tool_name}",
            {"tool_name": tool_name, "timeout_seconds": timeout_seconds},
        )
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        return _error("TOOL_ERROR", f"工具调用失败：{tool_name}", {"tool_name": tool_name})
    finally:
        if "data" in locals():
            executor.shutdown(wait=True)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    success = _tool_data_succeeded(data)
    return ToolResult(
        success=success,
        data=data,
        error_code=None if success else str(data.get("error_code") or data.get("error") or data.get("status") or "TOOL_FAILED"),
        error_message=None if success else str(data.get("error_message") or "Tool returned a failed status."),
        metadata={"tool_name": tool_name, "latency_ms": latency_ms},
    )


def _course_retriever(payload: BaseModel) -> dict[str, Any]:
    response = HybridRetriever(storage=LearningLogger()).retrieve(
        payload.query,
        top_k=payload.top_k,
        course_id=payload.course_id,
    )
    results = []
    for item in response.results:
        result = item.to_dict()
        result.update(
            {
                "score": item.final_score,
                "course_id": payload.course_id,
                "origin_kind": "course_document",
                "delivery_mode": "local_rag",
            }
        )
        results.append(result)
    return {
        "course_id": payload.course_id,
        "results": results,
        "abstained": response.abstained,
        "threshold": response.threshold,
        "fallback_used": response.fallback_used,
        "origin_kind": "course_document",
        "delivery_mode": "local_rag",
    }


def _paper_search(payload: BaseModel) -> dict[str, Any]:
    results = LocalRetriever().search(f"论文 研究 文献 {payload.query}", top_k=payload.top_k)
    local_clues = [
        {
            **item,
            "resource_type": "course_document",
            "verification_status": "local_course_clue",
            "url": "",
            "doi": "",
            "arxiv_id": "",
        }
        for item in results
    ]
    return {
        "results": local_clues,
        "verified_papers": [],
        "result_type": "local_course_clues",
        "note": "Local paper_search returns course clues only; it is not a verified paper database.",
    }


def _vuln_lookup(payload: BaseModel) -> dict[str, Any]:
    results = LocalRetriever().search(f"CVE 漏洞 PoC {payload.query}", top_k=payload.top_k)
    return {"results": results}


def _standard_lookup(payload: BaseModel) -> dict[str, Any]:
    results = LocalRetriever().search(f"3GPP TS 24.501 标准 {payload.query}", top_k=payload.top_k)
    return {"results": results}


def _paper_search_real(payload: BaseModel, *, deadline: float | None = None) -> dict[str, Any]:
    query = ResourceQuery(query=payload.query, resource_type="paper", page_size=payload.top_k)
    if payload.source == "auto":
        return _run_connector_search(None, query, deadline=deadline)
    return _run_connector_sync(payload.source, query, deadline=deadline)


def _vuln_lookup_real(payload: BaseModel, *, deadline: float | None = None) -> dict[str, Any]:
    query = ResourceQuery(query=payload.query, resource_type="vulnerability", page_size=payload.top_k)
    result = _run_connector_sync(
        "nvd",
        query,
        deadline=deadline,
    )
    return {**result, **nvd_query_metadata(query)}


def _standard_metadata_lookup(payload: BaseModel, *, deadline: float | None = None) -> dict[str, Any]:
    return _run_connector_sync(
        "standard_source",
        ResourceQuery(query=payload.query, resource_type="standard", page_size=payload.top_k),
        deadline=deadline,
    )


def _standard_lookup_real(payload: BaseModel, *, deadline: float | None = None) -> dict[str, Any]:
    return _run_connector_search(
        None,
        ResourceQuery(query=payload.query, resource_type="standard", page_size=payload.top_k),
        deadline=deadline,
    )


def _github_project_search(payload: BaseModel, *, deadline: float | None = None) -> dict[str, Any]:
    return _run_connector_sync(
        "github",
        ResourceQuery(query=payload.query, resource_type="project", page_size=payload.top_k),
        deadline=deadline,
    )


def _run_connector_sync(
    connector_name: str,
    query: ResourceQuery,
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    return _run_connector_search(connector_name, query, deadline=deadline)


def _run_connector_search(
    connector_name: str | None,
    query: ResourceQuery,
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    service = ExternalResourceService(
        storage=LearningLogger(),
        registry=get_default_registry(),
        vector_store=_NoopVectorStore(),
    )
    result = asyncio.run(service.search(query, connector_name=connector_name, deadline=deadline))
    return {
        "connector": result.get("connector"),
        "status": result.get("status", "FAILED"),
        "cache_hit": result.get("cache_hit", False),
        "error_code": result.get("error"),
        "error_message": result.get("error_message"),
        "upstream_error_code": result.get("upstream_error_code"),
        "upstream_error_message": result.get("upstream_error_message"),
        "delivery_mode": result.get("delivery_mode"),
        "origin_kind": result.get("origin_kind"),
        "fallback_used": result.get("fallback_used", False),
        "connectors_tried": result.get("connectors_tried", []),
        "lookup_mode": result.get("lookup_mode"),
        "canonical_identifier": result.get("canonical_identifier"),
        "lookup_state": result.get("lookup_state"),
        "source_attempts": result.get("source_attempts", []),
        "incomplete_results": result.get("incomplete_results"),
        "resources": result.get("resources") or [],
    }


def _tool_data_succeeded(data: dict[str, Any]) -> bool:
    status = str(data.get("status") or "").upper()
    if not status:
        return True
    if status in {"SUCCESS", "PARTIAL"}:
        return True
    if status == "FAILED_WITH_CACHE" and data.get("resources"):
        return True
    return False


class _NoopVectorStore:
    def upsert_external_resource(self, vector_id: str, text: str, record: dict[str, Any]) -> None:
        return None

    def delete_external_resource(self, vector_id: str) -> None:
        return None


def _mermaid_generator(payload: BaseModel) -> dict[str, Any]:
    if payload.diagram_type == "sequence":
        mermaid = (
            "sequenceDiagram\n"
            "    participant Student\n"
            "    participant Agent\n"
            "    participant Tool\n"
            f"    Student->>Agent: {payload.topic}\n"
            "    Agent->>Tool: retrieve evidence\n"
            "    Tool-->>Agent: observation\n"
            "    Agent-->>Student: grounded answer\n"
        )
    elif payload.diagram_type == "mindmap":
        mermaid = f"mindmap\n  root(({payload.topic}))\n    RAG\n    MCP Tools\n    Teacher Analytics\n"
    else:
        mermaid = (
            "flowchart TD\n"
            f"    A[学生问题: {payload.topic}] --> B[Agent Planner]\n"
            "    B --> C[Hybrid Retriever]\n"
            "    B --> D[MCP Tools]\n"
            "    C --> E[Evidence Bundle]\n"
            "    D --> E\n"
            "    E --> F[Grounded Answer]\n"
        )
    return {"mermaid": mermaid}


def _error(error_code: str, message: str, metadata: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=False,
        error_code=error_code,
        error_message=message,
        metadata=metadata,
    )
