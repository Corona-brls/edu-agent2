"""Semantic clustering for student questions and FAQ export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from backend.rag.hybrid_retriever import HybridRetriever
from backend.storage.logger import LearningLogger


class QuestionEmbedding(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class ClusterLabeler(Protocol):
    def label(self, questions: list[str]) -> str:
        ...


@dataclass(frozen=True)
class ClusterBuildOptions:
    course_id: int | None = None
    class_id: int | None = None
    class_ids: tuple[int, ...] | None = None
    start_at: str | None = None
    end_at: str | None = None
    distance_threshold: float = 0.45
    min_cluster_size: int = 2


class TfidfQuestionEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        normalized = [_normalize_question(text) for text in texts]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        matrix = vectorizer.fit_transform(normalized)
        return matrix.toarray().tolist()


def _normalize_question(text: str) -> str:
    lowered = " ".join(str(text or "").lower().split())
    canonical: list[str] = []
    synonym_groups = [
        ("nas", ["5g nas", "nas", "非接入层", "non-access stratum"]),
        ("authentication", ["认证", "鉴权", "身份验证", "接入认证", "authentication", "auth"]),
        ("process", ["流程", "过程", "步骤", "procedure", "process"]),
        ("risk", ["风险", "威胁", "问题", "安全隐患", "risk", "threat"]),
        ("replay", ["重放", "replay"]),
        ("blockchain", ["区块链", "blockchain"]),
        ("covert_channel", ["隐蔽通信", "隐蔽信道", "covert channel", "covert"]),
        ("vulnerability", ["漏洞", "cve", "vulnerability", "poc"]),
        ("standard", ["标准", "3gpp", "ts 24.501", "standard"]),
        ("experiment", ["实验", "实践", "experiment", "lab"]),
    ]
    for canonical_token, aliases in synonym_groups:
        if any(alias in lowered for alias in aliases):
            canonical.extend([canonical_token, canonical_token])
    return f"{lowered} {' '.join(canonical)}".strip()


class KeywordClusterLabeler:
    def label(self, questions: list[str]) -> str:
        text = " ".join(questions).lower()
        keyword_labels = [
            ("5g nas", "5G NAS"),
            ("nas", "NAS"),
            ("区块链", "区块链"),
            ("blockchain", "区块链"),
            ("隐蔽通信", "隐蔽通信"),
            ("covert", "隐蔽通信"),
            ("llm", "LLM"),
            ("漏洞", "漏洞"),
            ("cve", "CVE"),
            ("标准", "标准"),
            ("3gpp", "标准"),
            ("实验", "实验"),
        ]
        for keyword, label in keyword_labels:
            if keyword in text:
                return label
        first = questions[0] if questions else "常见问题"
        return first[:24]


class QuestionClusteringService:
    def __init__(
        self,
        storage: LearningLogger,
        *,
        embedding: QuestionEmbedding | None = None,
        labeler: ClusterLabeler | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.storage = storage
        self.embedding = embedding or TfidfQuestionEmbedding()
        self.labeler = labeler or KeywordClusterLabeler()
        self.retriever = retriever or HybridRetriever(storage=storage, vector_store=_KeywordOnlyVectorStore())

    def rebuild(self, options: ClusterBuildOptions | None = None) -> dict[str, Any]:
        options = options or ClusterBuildOptions()
        questions = self.storage.questions_for_clustering(
            course_id=options.course_id,
            class_id=options.class_id,
            class_ids=options.class_ids,
            start_at=options.start_at,
            end_at=options.end_at,
        )
        protected_ids = self.storage.protected_cluster_question_ids(
            course_id=options.course_id,
            class_id=options.class_id,
            class_ids=options.class_ids,
        )
        candidates = [item for item in questions if int(item["id"]) not in protected_ids]
        if not candidates:
            self.storage.replace_auto_question_clusters(
                [],
                course_id=options.course_id,
                class_id=options.class_id,
                class_ids=options.class_ids,
            )
            return {"clusters_created": 0, "questions_clustered": 0, "preserved_questions": len(protected_ids)}

        cluster_specs: list[dict[str, Any]] = []
        clustered = 0
        for bucket in _bucket_by_class(candidates):
            vectors = np.array(self.embedding.embed([item["question"] for item in bucket]), dtype=float)
            if vectors.ndim != 2 or vectors.shape[0] != len(bucket):
                raise ValueError("embedding returned invalid shape")
            distances = cosine_distances(vectors)
            labels = DBSCAN(
                eps=options.distance_threshold,
                min_samples=max(1, options.min_cluster_size),
                metric="precomputed",
            ).fit_predict(distances)

            grouped: dict[int, list[int]] = {}
            for index, label in enumerate(labels):
                group_label = -100000 - index if int(label) == -1 else int(label)
                grouped.setdefault(group_label, []).append(index)

            for label, indexes in grouped.items():
                rows = [bucket[index] for index in indexes]
                is_noise = label < 0 or len(rows) < options.min_cluster_size
                representative_index = _representative_index(distances, indexes)
                representative = bucket[representative_index]
                cluster_label = self._label_cluster([row["question"] for row in rows], is_noise=is_noise)
                cluster_specs.append(
                    {
                        "label": cluster_label,
                        "representative_question": representative["question"],
                        "course_id": representative.get("course_id"),
                        "class_id": representative.get("class_id"),
                        "approved": False,
                        "is_noise": is_noise,
                        "source": "auto",
                        "members": [
                            {
                                "question_id": int(bucket[index]["id"]),
                                "distance": round(float(distances[index, representative_index]), 4),
                                "is_representative": index == representative_index,
                            }
                            for index in indexes
                        ],
                    }
                )
                clustered += len(rows)
        created_ids = self.storage.replace_auto_question_clusters(
            cluster_specs,
            course_id=options.course_id,
            class_id=options.class_id,
            class_ids=options.class_ids,
        )
        return {"clusters_created": len(created_ids), "questions_clustered": clustered, "preserved_questions": len(protected_ids)}

    def list_clusters(
        self,
        course_id: int | None = None,
        *,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        return {"clusters": self.storage.question_clusters(course_id=course_id, class_id=class_id, class_ids=class_ids)}

    def rename(self, cluster_id: int, label: str) -> dict[str, Any]:
        self.storage.rename_question_cluster(cluster_id, label)
        return {"cluster_id": cluster_id, "label": label}

    def approve(self, cluster_id: int) -> dict[str, Any]:
        self.storage.approve_question_cluster(cluster_id)
        return {"cluster_id": cluster_id, "approved": True}

    def merge(self, target_cluster_id: int, source_cluster_ids: list[int], label: str | None = None) -> dict[str, Any]:
        self.storage.merge_question_clusters(target_cluster_id, source_cluster_ids, label)
        return {"target_cluster_id": target_cluster_id, "merged_cluster_ids": source_cluster_ids}

    def split(self, cluster_id: int, question_ids: list[int], label: str) -> dict[str, Any]:
        new_id = self.storage.split_question_cluster(cluster_id, question_ids, label)
        return {"source_cluster_id": cluster_id, "new_cluster_id": new_id}

    def export_faq(
        self,
        course_id: int | None = None,
        *,
        class_id: int | None = None,
        class_ids: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        items = []
        for cluster in self.storage.question_clusters(course_id=course_id, class_id=class_id, class_ids=class_ids):
            if cluster.get("is_noise") or not cluster.get("members") or not cluster.get("approved"):
                continue
            answer, citations = self._answer_with_citations(
                cluster["representative_question"],
                course_id=cluster.get("course_id") or course_id,
            )
            if not citations:
                continue
            items.append(
                {
                    "cluster_id": cluster["id"],
                    "label": cluster["label"],
                    "representative_question": cluster["representative_question"],
                    "count": cluster["count"],
                    "approved": cluster["approved"],
                    "answer": answer,
                    "citations": citations,
                }
            )
        return {"items": items}

    def _label_cluster(self, questions: list[str], *, is_noise: bool) -> str:
        if is_noise:
            return "噪声/小簇"
        try:
            label = self.labeler.label(questions).strip()
        except Exception:
            label = ""
        return label or KeywordClusterLabeler().label(questions)

    def _answer_with_citations(self, question: str, *, course_id: int | None = None) -> tuple[str, list[dict[str, Any]]]:
        try:
            results = self.retriever.retrieve(question, top_k=2, course_id=course_id).results
        except Exception:
            results = []
        citations = []
        for index, result in enumerate(results, 1):
            citations.append(
                {
                    "source_id": f"S{index}",
                    "source": result.source,
                    "title": result.title or result.source,
                    "snippet": result.snippet,
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "page": result.page,
                    "section": result.section,
                }
            )
        if not citations:
            return "本地知识库证据不足，建议教师补充资料后再发布该 FAQ。", []
        snippets = "；".join(f"[S{index}] {item['snippet']}" for index, item in enumerate(citations, 1))
        return f"根据本地知识库证据，可参考：{snippets}", citations


def _representative_index(distances: np.ndarray, indexes: list[int]) -> int:
    if len(indexes) == 1:
        return indexes[0]
    best = min(indexes, key=lambda index: float(distances[index, indexes].mean()))
    return int(best)


def _bucket_by_class(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    buckets: dict[tuple[int | None, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("course_id"), row.get("class_id"))
        buckets.setdefault(key, []).append(row)
    return [buckets[key] for key in sorted(buckets, key=lambda item: ((item[0] or 0), (item[1] or 0)))]


class _KeywordOnlyVectorStore:
    def query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise RuntimeError("question clustering FAQ uses keyword-only RAG fallback by default")
