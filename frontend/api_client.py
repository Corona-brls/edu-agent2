"""HTTP client helpers for the Streamlit frontend."""

from __future__ import annotations

import os
from typing import Any

import requests

from frontend.presentation import safe_course_reference_status, safe_error_message, safe_request_id

API_BASE_URL = os.getenv("COURSE_AGENT_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 60


class APIClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        retry_after: int | None = None,
        course_reference_status: str | None = None,
    ) -> None:
        # The caller-supplied message can contain upstream bodies, URLs or secrets.
        # Keep structured fields, but never expose that untrusted text via str(exc).
        super().__init__(safe_error_message(
            status_code=status_code,
            error_code=error_code,
            retry_after=retry_after,
            request_id=request_id,
            course_reference_status=course_reference_status,
        ))
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = safe_request_id(request_id)
        self.retryable = retryable
        self.retry_after = retry_after
        self.course_reference_status = safe_course_reference_status(course_reference_status)


def health() -> dict[str, Any]:
    return _request("GET", "/api/health")


def system_status() -> dict[str, Any]:
    return _request("GET", "/api/system/status")


def login(username: str, password: str) -> dict[str, Any]:
    return _request("POST", "/api/auth/login", json={"username": username, "password": password})


def me(token: str) -> dict[str, Any]:
    return _request("GET", "/api/auth/me", headers=_auth_headers(token))


def courses(token: str) -> list[dict[str, Any]]:
    return _request("GET", "/api/courses", headers=_auth_headers(token))


def classes(token: str, *, course_id: int) -> list[dict[str, Any]]:
    return _request(
        "GET",
        "/api/classes",
        headers=_auth_headers(token),
        params={"course_id": course_id},
    )


def logout(token: str) -> dict[str, Any]:
    return _request("POST", "/api/auth/logout", headers=_auth_headers(token))


def change_password(token: str, current_password: str, new_password: str) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/auth/change-password",
        headers=_auth_headers(token),
        json={"current_password": current_password, "new_password": new_password},
    )


def create_learning_session(
    token: str,
    client_session_id: str,
    *,
    course_id: int,
    class_id: int | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/learning/sessions",
        headers=_auth_headers(token),
        json={"client_session_id": client_session_id, "course_id": course_id, "class_id": class_id},
    )


def heartbeat_learning_session(token: str, client_session_id: str) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/learning/sessions/{client_session_id}/heartbeat",
        headers=_auth_headers(token),
    )


def end_learning_session(token: str, client_session_id: str) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/learning/sessions/{client_session_id}/end",
        headers=_auth_headers(token),
    )


def record_learning_event(
    token: str,
    client_session_id: str,
    event_type: str,
    page_name: str = "",
    module: str = "",
    assignment_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/learning/events",
        headers=_auth_headers(token),
        json={
            "client_session_id": client_session_id,
            "event_type": event_type,
            "page_name": page_name,
            "module": module,
            "assignment_id": assignment_id,
            "metadata": metadata or {},
            "dedupe_key": dedupe_key,
            "course_id": course_id,
            "class_id": class_id,
        },
    )


def general_chat(
    token: str,
    question: str,
    *,
    course_id: int,
    class_id: int | None = None,
    use_course_references: bool = False,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "course_id": course_id,
        "class_id": class_id,
        "use_course_references": use_course_references,
    }
    if client_session_id:
        payload["client_session_id"] = client_session_id
    return _request(
        "POST",
        "/api/learning/general-chat",
        headers=_auth_headers(token),
        json=payload,
        timeout=105,
    )


def teacher_activity(token: str) -> dict[str, Any]:
    return _request("GET", "/api/teacher/activity", headers=_auth_headers(token))


def teacher_phase3_summary(
    token: str,
    *,
    course_id: int | None = None,
    class_id: int | None = None,
    start_at: str | None = None,
    end_at: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if course_id is not None:
        params["course_id"] = course_id
    if class_id is not None:
        params["class_id"] = class_id
    if start_at:
        params["start_at"] = start_at
    if end_at:
        params["end_at"] = end_at
    return _request("GET", "/api/teacher/phase3/summary", headers=_auth_headers(token), params=params)


def teacher_phase3_student_timeline(token: str, student_id: int, *, limit: int = 100) -> dict[str, Any]:
    return _request(
        "GET",
        f"/api/teacher/phase3/students/{student_id}/timeline",
        headers=_auth_headers(token),
        params={"limit": limit},
    )


def teacher_question_cluster_faq_json(
    token: str,
    *,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if course_id is not None:
        params["course_id"] = course_id
    if class_id is not None:
        params["class_id"] = class_id
    return _request(
        "GET",
        "/api/teacher/question-clusters/faq-export",
        headers=_auth_headers(token),
        params=params or None,
    )


def teacher_phase3_faq_csv(token: str, *, course_id: int | None = None, class_id: int | None = None) -> bytes:
    params: dict[str, Any] = {}
    if course_id is not None:
        params["course_id"] = course_id
    if class_id is not None:
        params["class_id"] = class_id
    try:
        response = requests.request(
            "GET",
            f"{API_BASE_URL}/api/teacher/phase3/faq-export.csv",
            timeout=TIMEOUT_SECONDS,
            headers=_auth_headers(token),
            params=params or None,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        details = _extract_error_details(exc.response)
        raise APIClientError(
            details["message"],
            status_code=exc.response.status_code if exc.response is not None else None,
            error_code=details.get("error_code"),
            request_id=details.get("request_id"),
            retryable=details.get("retryable"),
            retry_after=details.get("retry_after"),
            course_reference_status=details.get("course_reference_status"),
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError("网络请求失败") from exc
    return response.content


def approve_question_cluster(token: str, cluster_id: int) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/teacher/question-clusters/{cluster_id}/approve",
        headers=_auth_headers(token),
    )


def rename_question_cluster(token: str, cluster_id: int, label: str) -> dict[str, Any]:
    return _request(
        "PATCH",
        f"/api/teacher/question-clusters/{cluster_id}",
        headers=_auth_headers(token),
        json={"label": label},
    )


def merge_question_clusters(
    token: str,
    *,
    target_cluster_id: int,
    source_cluster_ids: list[int],
    label: str | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/teacher/question-clusters/merge",
        headers=_auth_headers(token),
        json={
            "target_cluster_id": target_cluster_id,
            "source_cluster_ids": source_cluster_ids,
            "label": label or None,
        },
    )


def split_question_cluster(token: str, cluster_id: int, *, question_ids: list[int], label: str) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/teacher/question-clusters/{cluster_id}/split",
        headers=_auth_headers(token),
        json={"question_ids": question_ids, "label": label},
    )


def student_timeline(token: str, student_id: int) -> dict[str, Any]:
    return _request("GET", f"/api/students/{student_id}/timeline", headers=_auth_headers(token))


def chat(
    question: str,
    user_id: str = "demo_student",
    conversation_id: str | None = None,
    token: str | None = None,
    client_session_id: str | None = None,
    assignment_id: int | None = None,
    step_id: int | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
    use_course_references: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "user_id": user_id,
        "use_course_references": use_course_references,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if client_session_id:
        payload["client_session_id"] = client_session_id
    if assignment_id is not None:
        payload["assignment_id"] = assignment_id
    if step_id is not None:
        payload["step_id"] = step_id
    if course_id is not None:
        payload["course_id"] = course_id
    if class_id is not None:
        payload["class_id"] = class_id
    return _request("POST", "/api/chat", json=payload, headers=_auth_headers(token) if token else None)


def continue_agent_run(
    run_id: str,
    user_input: str,
    user_id: str = "demo_student",
    conversation_id: str | None = None,
    token: str | None = None,
    client_session_id: str | None = None,
    assignment_id: int | None = None,
    step_id: int | None = None,
    use_course_references: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_input": user_input,
        "user_id": user_id,
        "use_course_references": use_course_references,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if client_session_id:
        payload["client_session_id"] = client_session_id
    if assignment_id is not None:
        payload["assignment_id"] = assignment_id
    if step_id is not None:
        payload["step_id"] = step_id
    return _request(
        "POST",
        f"/api/agent/runs/{run_id}/continue",
        json=payload,
        headers=_auth_headers(token) if token else None,
    )


def get_agent_run(run_id: str, token: str | None = None) -> dict[str, Any]:
    return _request("GET", f"/api/agent/runs/{run_id}", headers=_auth_headers(token) if token else None)


def practice_guide(
    topic: str,
    user_id: str = "demo_student",
    token: str | None = None,
    client_session_id: str | None = None,
    assignment_id: int | None = None,
    step_id: int | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
    use_course_references: bool = False,
) -> dict[str, Any]:
    payload = {
        "topic": topic,
        "user_id": user_id,
        "use_course_references": use_course_references,
    }
    if client_session_id:
        payload["client_session_id"] = client_session_id
    if assignment_id is not None:
        payload["assignment_id"] = assignment_id
    if step_id is not None:
        payload["step_id"] = step_id
    if course_id is not None:
        payload["course_id"] = course_id
    if class_id is not None:
        payload["class_id"] = class_id
    return _request("POST", "/api/practice/guide", json=payload, headers=_auth_headers(token) if token else None)


def practice_draft(
    token: str,
    *,
    topic: str,
    objective: str,
    lab_environment: str,
    available_tools: list[str],
    course_id: int,
    class_id: int | None,
    use_course_references: bool = False,
    external_queries: list[dict[str, str]] | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "topic": topic,
        "objective": objective,
        "lab_environment": lab_environment,
        "available_tools": available_tools,
        "course_id": course_id,
        "class_id": class_id,
        "use_course_references": use_course_references,
        "external_queries": external_queries or [],
    }
    if client_session_id:
        payload["client_session_id"] = client_session_id
    return _request(
        "POST",
        "/api/practice/drafts",
        json=payload,
        headers=_auth_headers(token),
        timeout=105,
    )


def assignments(
    token: str,
    *,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if course_id is not None:
        params["course_id"] = course_id
    if class_id is not None:
        params["class_id"] = class_id
    return _request("GET", "/api/assignments", headers=_auth_headers(token), params=params or None)


def assignment_progress(token: str, assignment_id: int) -> dict[str, Any]:
    return _request("GET", f"/api/assignments/{assignment_id}/progress", headers=_auth_headers(token))


def update_assignment_progress(
    token: str,
    assignment_id: int,
    *,
    step_id: int,
    status: str,
    progress_percent: int,
    note: str = "",
    client_session_id: str | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/assignments/{assignment_id}/progress",
        headers=_auth_headers(token),
        json={
            "step_id": step_id,
            "status": status,
            "progress_percent": progress_percent,
            "note": note,
            "client_session_id": client_session_id,
        },
    )


def report_stuck(
    token: str,
    *,
    assignment_id: int,
    step_id: int,
    reason: str,
    module: str = "",
    client_session_id: str | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/stuck-reports",
        headers=_auth_headers(token),
        json={
            "assignment_id": assignment_id,
            "step_id": step_id,
            "reason": reason,
            "module": module,
            "client_session_id": client_session_id,
        },
    )


def teacher_stuck_analysis(
    token: str,
    *,
    assignment_id: int | None = None,
    step_id: int | None = None,
    module: str = "",
    class_id: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"module": module}
    if assignment_id is not None:
        params["assignment_id"] = assignment_id
    if step_id is not None:
        params["step_id"] = step_id
    if class_id is not None:
        params["class_id"] = class_id
    return _request("GET", "/api/teacher/stuck-analysis", headers=_auth_headers(token), params=params)


def update_stuck_report(token: str, report_id: int, action: str) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/teacher/stuck-reports/{report_id}/action",
        headers=_auth_headers(token),
        json={"action": action},
    )


def get_tools() -> list[dict[str, Any]]:
    return _request("GET", "/api/tools")


def connectors_status() -> dict[str, Any]:
    return _request("GET", "/api/connectors/status")


def sync_connector(
    token: str,
    name: str,
    *,
    query: str,
    resource_type: str | None = None,
    page_size: int = 5,
    module: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "page_size": page_size,
        "module": module,
        "allow_remote": True,
    }
    if resource_type:
        payload["resource_type"] = resource_type
    return _request(
        "POST",
        f"/api/connectors/{name}/sync",
        headers=_require_auth_headers(token),
        json=payload,
    )


def connector_sync_runs(token: str, limit: int = 20) -> dict[str, Any]:
    return _request(
        "GET",
        "/api/connectors/sync-runs",
        headers=_require_auth_headers(token),
        params={"limit": limit},
    )


def teacher_analytics(token: str | None = None) -> dict[str, Any]:
    return _request("GET", "/api/teacher/analytics", headers=_require_auth_headers(token))


def teacher_faq(token: str | None = None) -> dict[str, Any]:
    return _request("GET", "/api/teacher/faq", headers=_require_auth_headers(token))


def resource_portal(
    token: str,
    *,
    keyword: str = "",
    topic: str = "",
    resource_type: str = "",
    year_from: int | None = None,
    year_to: int | None = None,
    source: str = "",
    page: int = 1,
    page_size: int = 10,
    client_session_id: str | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "keyword": keyword,
        "topic": topic,
        "resource_type": resource_type,
        "source": source,
        "page": page,
        "page_size": page_size,
    }
    if year_from is not None:
        params["year_from"] = year_from
    if year_to is not None:
        params["year_to"] = year_to
    if client_session_id:
        params["client_session_id"] = client_session_id
    if course_id is not None:
        params["course_id"] = course_id
    if class_id is not None:
        params["class_id"] = class_id
    return _request("GET", "/api/resources/portal", headers=_auth_headers(token), params=params)


def search_resources(
    token: str,
    *,
    query: str = "",
    resource_type: str = "",
    module: str = "",
    page: int = 1,
    page_size: int = 10,
    connector_name: str | None = None,
) -> dict[str, Any]:
    params = {"connector_name": connector_name} if connector_name else None
    return _request(
        "POST",
        "/api/resources/search",
        headers=_auth_headers(token),
        params=params,
        json={
            "query": query,
            "resource_type": resource_type or None,
            "module": module,
            "page": page,
            "page_size": page_size,
            "allow_remote": True,
        },
    )


def open_resource(
    token: str,
    resource_id: str,
    *,
    client_session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/resources/open",
        headers=_auth_headers(token),
        json={
            "resource_id": resource_id,
            "client_session_id": client_session_id,
            "course_id": course_id,
            "class_id": class_id,
            "metadata": metadata or {},
        },
    )


def start_innovation_session(
    token: str,
    *,
    topic: str,
    current_plan: str,
    client_session_id: str | None = None,
    course_id: int | None = None,
    class_id: int | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/innovation/sessions",
        headers=_auth_headers(token),
        json={
            "topic": topic,
            "current_plan": current_plan,
            "client_session_id": client_session_id,
            "course_id": course_id,
            "class_id": class_id,
        },
    )


def list_innovation_sessions(token: str, *, course_id: int, class_id: int | None) -> dict[str, Any]:
    return _request(
        "GET", "/api/innovation/sessions", headers=_auth_headers(token),
        params={"course_id": course_id, "class_id": class_id},
    )


def get_innovation_session(token: str, session_id: int) -> dict[str, Any]:
    return _request("GET", f"/api/innovation/sessions/{session_id}", headers=_auth_headers(token))


def answer_innovation_question(
    token: str,
    session_id: int,
    *,
    question_id: int,
    answer: str,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/innovation/sessions/{session_id}/answers",
        headers=_auth_headers(token),
        json={
            "question_id": question_id,
            "answer": answer,
            "client_session_id": client_session_id,
        },
    )


def finalize_innovation_session(
    token: str,
    session_id: int,
    *,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/innovation/sessions/{session_id}/finalize",
        headers=_auth_headers(token),
        json={"client_session_id": client_session_id},
    )


def innovation_suggestions(
    token: str,
    session_id: int,
    *,
    client_session_id: str | None = None,
    use_course_references: bool = False,
) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/innovation/sessions/{session_id}/suggestions",
        headers=_auth_headers(token),
        json={
            "client_session_id": client_session_id,
            "use_course_references": bool(use_course_references),
        },
        timeout=105.0,
    )


def upload_knowledge(
    filename: str,
    content: bytes,
    token: str | None = None,
    *,
    course_id: int | None = None,
) -> dict[str, Any]:
    files = {"file": (filename, content)}
    params = {"course_id": course_id} if course_id is not None else None
    return _request(
        "POST",
        "/api/knowledge/upload",
        headers=_require_auth_headers(token),
        params=params,
        files=files,
    )


def knowledge_documents(token: str | None = None, *, course_id: int | None = None) -> dict[str, Any]:
    params = {"course_id": course_id} if course_id is not None else None
    return _request(
        "GET",
        "/api/knowledge/documents",
        headers=_require_auth_headers(token),
        params=params,
    )


def reindex_document(document_id: int, token: str | None = None) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/knowledge/documents/{document_id}/reindex",
        headers=_require_auth_headers(token),
    )


def publish_document(document_id: int, token: str | None = None) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/knowledge/documents/{document_id}/publish",
        headers=_require_auth_headers(token),
    )


def unpublish_document(document_id: int, token: str | None = None) -> dict[str, Any]:
    return _request(
        "POST",
        f"/api/knowledge/documents/{document_id}/unpublish",
        headers=_require_auth_headers(token),
    )


def delete_document(document_id: int, token: str | None = None) -> dict[str, Any]:
    return _request(
        "DELETE",
        f"/api/knowledge/documents/{document_id}",
        headers=_require_auth_headers(token),
    )


def submit_feedback(
    question_id: int,
    rating: int,
    helpful: bool | None = None,
    comment: str = "",
    user_id: str = "demo_student",
    token: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "question_id": question_id,
        "rating": rating,
        "helpful": helpful,
        "comment": comment,
        "user_id": user_id,
    }
    if client_session_id:
        payload["client_session_id"] = client_session_id
    return _request(
        "POST",
        "/api/feedback",
        json=payload,
        headers=_auth_headers(token) if token else None,
    )


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{API_BASE_URL}{path}"
    timeout = kwargs.pop("timeout", TIMEOUT_SECONDS)
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
    except requests.HTTPError as exc:
        details = _extract_error_details(exc.response)
        raise APIClientError(
            f"后端接口返回错误：{details['message']}",
            status_code=exc.response.status_code if exc.response is not None else None,
            error_code=details.get("error_code"),
            request_id=details.get("request_id"),
            retryable=details.get("retryable"),
            retry_after=details.get("retry_after"),
            course_reference_status=details.get("course_reference_status"),
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError("网络请求失败") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("响应格式错误", status_code=response.status_code) from exc


def _extract_error_message(response: requests.Response | None) -> str:
    return str(_extract_error_details(response)["message"])


def _extract_error_details(response: requests.Response | None) -> dict[str, Any]:
    if response is None:
        return {"message": safe_error_message()}
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    detail = payload.get("detail")
    if isinstance(detail, dict):
        source = {**detail, **payload}
    else:
        source = payload
    retry_after = response.headers.get("Retry-After")
    try:
        parsed_retry_after = int(retry_after) if retry_after else None
    except (TypeError, ValueError):
        parsed_retry_after = None
    if parsed_retry_after is not None and not 0 < parsed_retry_after <= 86400:
        parsed_retry_after = None
    error_code = source.get("error_code")
    request_id = safe_request_id(source.get("request_id"))
    course_reference_status = safe_course_reference_status(source.get("course_reference_status"))
    return {
        "message": safe_error_message(
            status_code=response.status_code,
            error_code=error_code,
            retry_after=parsed_retry_after,
            request_id=request_id,
            course_reference_status=course_reference_status,
        ),
        "error_code": error_code,
        "request_id": request_id,
        "retryable": source.get("retryable") if isinstance(source.get("retryable"), bool) else None,
        "retry_after": parsed_retry_after,
        "course_reference_status": course_reference_status,
    }


def _require_auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        raise APIClientError("未登录", status_code=401)
    return _auth_headers(token)


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
