"""Chroma vector store for parsed document chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from backend.core.config import get_settings


class EmbeddingFunction(Protocol):
    def __call__(self, input: list[str]) -> list[list[float]]:
        ...


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
        import chromadb

        settings = get_settings()
        persist_dir = persist_dir or settings.vector_store_dir
        collection_name = collection_name or settings.vector_collection_name
        embedding_model = embedding_model or settings.embedding_model
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function or _sentence_transformer_embedding(embedding_model)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        prepared = [_prepare_chunk(row) for row in chunks if row.get("chunk_text")]
        if not prepared:
            return 0

        self.collection.upsert(
            ids=[item["id"] for item in prepared],
            documents=[item["text"] for item in prepared],
            metadatas=[item["metadata"] for item in prepared],
        )
        return len(prepared)

    def index_from_sqlite(self, storage: Any) -> int:
        chunks = storage.document_chunks(only_pending=True)
        inserted = self.add_chunks(chunks)
        if inserted:
            storage.mark_document_chunks_indexed([int(row["id"]) for row in chunks])
        return inserted

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0 or not text.strip():
            return []

        result = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[dict[str, Any]] = []
        for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            matches.append(
                {
                    "id": item_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": float(distance),
                    "score": _similarity_score(float(distance)),
                }
            )
        return matches

    def delete_by_document(self, document_id: int) -> None:
        self.collection.delete(where={"document_id": int(document_id)})

    def contains_chunks(self, chunks: list[dict[str, Any]]) -> bool:
        """Return whether every SQLite chunk identity is present in Chroma."""

        expected = {
            str(int(row["id"])): (int(row["document_id"]), int(row["id"]))
            for row in chunks
        }
        if not expected:
            return False
        result = self.collection.get(ids=list(expected), include=["metadatas"])
        ids = [str(value) for value in result.get("ids", [])]
        metadatas = result.get("metadatas", [])
        actual: dict[str, tuple[int, int]] = {}
        for item_id, metadata in zip(ids, metadatas):
            values = metadata or {}
            try:
                actual[item_id] = (int(values["document_id"]), int(values["chunk_id"]))
            except (KeyError, TypeError, ValueError):
                return False
        return actual == expected

    def upsert_external_resource(
        self,
        vector_id: str,
        text: str,
        record: dict[str, Any],
    ) -> None:
        self.collection.upsert(
            ids=[vector_id],
            documents=[text],
            metadatas=[
                {
                    "external_resource_id": int(record["id"]),
                    "source_name": record.get("source_name", ""),
                    "external_id": record.get("external_id", ""),
                    "resource_type": record.get("resource_type", ""),
                    "title": record.get("title", ""),
                }
            ],
        )

    def delete_external_resource(self, vector_id: str) -> None:
        self.collection.delete(ids=[vector_id])

    def count(self) -> int:
        return int(self.collection.count())

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        self.reset_collection()


def _sentence_transformer_embedding(model_name: str):
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    offline = os.getenv("EMBEDDING_OFFLINE", "true").strip().lower() in {"1", "true", "yes", "on"}
    kwargs = {"local_files_only": True} if offline else {}
    return SentenceTransformerEmbeddingFunction(model_name=model_name, **kwargs)


def _similarity_score(distance: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - distance)), 4)


def _prepare_chunk(row: dict[str, Any]) -> dict[str, Any]:
    chunk_id = int(row["id"])
    document_id = int(row["document_id"])
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": int(row["chunk_index"]),
        "chunk_hash": row.get("chunk_hash", ""),
        "filename": row.get("filename", ""),
        "title": row.get("title", ""),
        "source_path": row.get("source_path", ""),
    }
    return {
        "id": str(chunk_id),
        "text": row["chunk_text"],
        "metadata": metadata,
    }
