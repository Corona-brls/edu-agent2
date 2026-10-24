"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path



def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_load_dotenv_if_available()


@dataclass(frozen=True)
class Settings:
    knowledge_base_dir: Path
    vector_store_dir: Path
    embedding_model: str
    vector_collection_name: str
    retrieval_top_k: int
    retrieval_keyword_weight: float
    retrieval_vector_weight: float
    retrieval_min_score: float
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    llm_temperature: float
    llm_max_output_tokens: int
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_fallback_provider: str
    mcp_tool_timeout_seconds: float
    agent_max_steps: int
    agent_max_tool_calls: int
    agent_timeout_seconds: float
    agent_allow_ask_user: bool
    jwt_algorithm: str
    access_token_expire_minutes: int
    enable_demo_accounts: bool
    default_course_id: str
    default_class_id: str
    session_idle_timeout_minutes: int
    activity_heartbeat_seconds: int
    learning_event_retention_days: int
    external_cache_ttl_hours: int
    external_http_timeout_seconds: float
    external_http_max_retries: int
    external_resource_index_enabled: bool
    enable_semantic_scholar: bool
    semantic_scholar_api_key: str
    enable_arxiv: bool
    enable_nvd: bool
    nvd_api_key: str
    enable_github_search: bool
    github_token: str
    enable_real_rfc_lookup: bool
    enable_standard_remote_metadata: bool
    question_cluster_auto_threshold: int
    question_cluster_distance_threshold: float
    question_cluster_min_size: int
    external_model_course_allowlist: frozenset[int]


def get_settings() -> Settings:
    return Settings(
        knowledge_base_dir=Path(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")).expanduser().resolve(),
        vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", "backend/data/vector_store")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        vector_collection_name=os.getenv("VECTOR_COLLECTION_NAME", "course_document_chunks"),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        retrieval_keyword_weight=float(os.getenv("RETRIEVAL_KEYWORD_WEIGHT", "0.45")),
        retrieval_vector_weight=float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.55")),
        retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.18")),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek").strip().lower(),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip(),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2000")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        llm_fallback_provider=os.getenv("LLM_FALLBACK_PROVIDER", "template").strip().lower(),
        mcp_tool_timeout_seconds=_env_bounded_float(
            "MCP_TOOL_TIMEOUT_SECONDS",
            default=30.0,
            minimum=0.05,
            maximum=300.0,
        ),
        agent_max_steps=int(os.getenv("AGENT_MAX_STEPS", "5")),
        agent_max_tool_calls=int(os.getenv("AGENT_MAX_TOOL_CALLS", "3")),
        agent_timeout_seconds=float(os.getenv("AGENT_TIMEOUT_SECONDS", "120")),
        agent_allow_ask_user=_env_bool("AGENT_ALLOW_ASK_USER", default=True),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256").strip(),
        access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480")),
        enable_demo_accounts=_env_bool("ENABLE_DEMO_ACCOUNTS", default=True),
        default_course_id=os.getenv("DEFAULT_COURSE_ID", "demo-course").strip(),
        default_class_id=os.getenv("DEFAULT_CLASS_ID", "demo-class").strip(),
        session_idle_timeout_minutes=int(os.getenv("SESSION_IDLE_TIMEOUT_MINUTES", "30")),
        activity_heartbeat_seconds=int(os.getenv("ACTIVITY_HEARTBEAT_SECONDS", "60")),
        learning_event_retention_days=int(os.getenv("LEARNING_EVENT_RETENTION_DAYS", "365")),
        external_cache_ttl_hours=int(os.getenv("EXTERNAL_CACHE_TTL_HOURS", "24")),
        external_http_timeout_seconds=float(os.getenv("EXTERNAL_HTTP_TIMEOUT_SECONDS", "20")),
        external_http_max_retries=int(os.getenv("EXTERNAL_HTTP_MAX_RETRIES", "2")),
        external_resource_index_enabled=_env_bool("EXTERNAL_RESOURCE_INDEX_ENABLED", default=False),
        enable_semantic_scholar=_env_bool("ENABLE_SEMANTIC_SCHOLAR", default=False),
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip(),
        enable_arxiv=_env_bool("ENABLE_ARXIV", default=False),
        enable_nvd=_env_bool("ENABLE_NVD", default=False),
        nvd_api_key=os.getenv("NVD_API_KEY", "").strip(),
        enable_github_search=_env_bool("ENABLE_GITHUB_SEARCH", default=False),
        github_token=os.getenv("GITHUB_TOKEN", "").strip(),
        enable_real_rfc_lookup=_env_bool("ENABLE_REAL_RFC_LOOKUP", default=False),
        enable_standard_remote_metadata=_env_bool("ENABLE_STANDARD_REMOTE_METADATA", default=True),
        question_cluster_auto_threshold=int(os.getenv("QUESTION_CLUSTER_AUTO_THRESHOLD", "3")),
        question_cluster_distance_threshold=float(os.getenv("QUESTION_CLUSTER_DISTANCE_THRESHOLD", "0.45")),
        question_cluster_min_size=int(os.getenv("QUESTION_CLUSTER_MIN_SIZE", "2")),
        external_model_course_allowlist=_parse_course_allowlist(
            os.getenv("EXTERNAL_MODEL_COURSE_ALLOWLIST", "")
        ),
    )


def _parse_course_allowlist(raw: str) -> frozenset[int]:
    values: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        if not text.isdigit() or int(text) <= 0:
            raise ValueError("EXTERNAL_MODEL_COURSE_ALLOWLIST must contain positive integer course IDs")
        values.add(int(text))
    return frozenset(values)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_bounded_float(
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name)
    value = default if raw is None or not raw.strip() else float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} seconds")
    return value
