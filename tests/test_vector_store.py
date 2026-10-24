from pathlib import Path

from backend.rag.document_parser import Document
from backend.rag.vector_store import ChromaVectorStore
from backend.storage.logger import LearningLogger, sha256_text


class KeywordEmbedding:
    def name(self) -> str:
        return "default"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("nas")),
            float(lowered.count("auth")),
            float(lowered.count("blockchain")),
            float(lowered.count("covert")),
            float(lowered.count("vulnerability")),
        ]


def test_chroma_vector_store_indexes_and_queries_sqlite_chunks(tmp_path):
    storage = LearningLogger(tmp_path / "test.db")
    storage.seed_demo_identity()
    course_id = storage.default_course_id()
    document = Document(
        source_path=Path("knowledge_base/sample.md"),
        filename="sample.md",
        file_type="md",
        title="Sample Course Notes",
        text="5G NAS authentication risk\nBlockchain covert channel design",
    )
    record = storage.save_document(
        document=document,
        content_hash=sha256_text(document.text),
        chunks=[
            "5G NAS authentication risk and replay vulnerability",
            "Blockchain covert channel design and transaction timing",
        ],
        byte_size=len(document.text.encode("utf-8")),
        course_id=course_id,
    )

    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_document_chunks",
        embedding_function=KeywordEmbedding(),
    )
    inserted = store.index_from_sqlite(storage)
    inserted_again = store.index_from_sqlite(storage)
    matches = store.query("NAS authentication vulnerability", top_k=1)
    empty_matches = store.query("NAS authentication vulnerability", top_k=0)
    chunks = storage.chunks_for_document(record["id"])

    assert inserted == 2
    assert inserted_again == 0
    assert store.count() == 2
    assert len(matches) == 1
    assert empty_matches == []
    assert "NAS authentication" in matches[0]["text"]
    assert matches[0]["id"] == str(matches[0]["metadata"]["chunk_id"])
    assert 0.0 <= matches[0]["score"] <= 1.0
    assert matches[0]["metadata"]["document_id"] == record["id"]
    assert {chunk["index_status"] for chunk in chunks} == {"INDEXED"}

    reopened = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_document_chunks",
        embedding_function=KeywordEmbedding(),
    )
    reopened_matches = reopened.query("Blockchain covert channel", top_k=1)

    assert reopened.count() == 2
    assert "Blockchain covert" in reopened_matches[0]["text"]

    reopened.delete_by_document(record["id"])

    assert reopened.count() == 0


def test_chroma_vector_store_can_insert_chunks_directly(tmp_path):
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="direct_chunks",
        embedding_function=KeywordEmbedding(),
    )

    inserted = store.add_chunks(
        [
            {
                "id": 10,
                "document_id": 1,
                "chunk_index": 0,
                "chunk_text": "NAS authentication procedure",
                "chunk_hash": "hash-a",
                "filename": "a.md",
                "title": "A",
                "source_path": "a.md",
            }
        ]
    )
    matches = store.query("NAS auth", top_k=1)

    assert inserted == 1
    assert matches[0]["id"] == "10"
    assert matches[0]["metadata"]["chunk_id"] == 10
    assert "NAS authentication" in matches[0]["text"]


def test_chroma_vector_store_reset_collection(tmp_path):
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="reset_chunks",
        embedding_function=KeywordEmbedding(),
    )
    store.add_chunks(
        [
            {
                "id": 20,
                "document_id": 2,
                "chunk_index": 0,
                "chunk_text": "Blockchain covert channel",
                "chunk_hash": "hash-b",
                "filename": "b.md",
                "title": "B",
                "source_path": "b.md",
            }
        ]
    )

    assert store.count() == 1

    store.reset_collection()

    assert store.count() == 0


def test_chroma_vector_store_reads_default_config_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "configured_chroma"))
    monkeypatch.setenv("VECTOR_COLLECTION_NAME", "configured_chunks")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")

    store = ChromaVectorStore(embedding_function=KeywordEmbedding())

    assert store.persist_dir == tmp_path / "configured_chroma"
    assert store.collection_name == "configured_chunks"
    assert store.embedding_model == "test-embedding-model"
