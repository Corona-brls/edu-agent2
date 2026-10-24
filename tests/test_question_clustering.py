from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend import main
from backend.analytics.clustering import QuestionClusteringService
from backend.rag.document_parser import Document, Page
from backend.storage.logger import LearningLogger, sha256_bytes


class FixedEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "NAS" in text or "认证" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "区块链" in text or "隐蔽" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "LLM" in text or "漏洞" in text:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.2, 0.2, 0.9])
        return vectors


class MockLabeler:
    def __init__(self, label: str = "NAS 安全问题") -> None:
        self._label = label

    def label(self, questions: list[str]) -> str:
        return self._label


class FailingLabeler:
    def label(self, questions: list[str]) -> str:
        raise RuntimeError("DeepSeek unavailable")


class RecordingFaqRetriever:
    def __init__(self) -> None:
        self.course_ids: list[int | None] = []

    def retrieve(self, question: str, top_k: int = 2, course_id: int | None = None):
        self.course_ids.append(course_id)

        class Result:
            source = "course-a.md"
            title = "Course A Evidence"
            snippet = "5G NAS authentication evidence."
            document_id = 1
            chunk_id = 1
            page = 1
            section = "faq"

        class Response:
            results = [Result()]

        return Response()


def _client(tmp_path, monkeypatch, *, labeler=None) -> TestClient:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-question-clustering-suite")
    original_logger = main.learning_logger
    original_factory = main.question_clustering_service_factory
    main.learning_logger = LearningLogger(tmp_path / "clusters.db")
    main.learning_logger.seed_demo_identity()
    _seed_questions_and_knowledge(main.learning_logger, tmp_path)

    def factory(storage):
        return QuestionClusteringService(storage, embedding=FixedEmbedding(), labeler=labeler or MockLabeler())

    main.question_clustering_service_factory = factory
    client = TestClient(main.app)
    client._original_logger = original_logger  # type: ignore[attr-defined]
    client._original_factory = original_factory  # type: ignore[attr-defined]
    return client


def _restore(client: TestClient) -> None:
    main.learning_logger = client._original_logger  # type: ignore[attr-defined]
    main.question_clustering_service_factory = client._original_factory  # type: ignore[attr-defined]


def _login(client: TestClient, username: str = "teacher_demo") -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": "demo123456"})
    assert response.status_code == 200
    payload = response.json()
    token = payload["access_token"]
    if payload.get("must_change_password"):
        changed = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "demo123456", "new_password": f"{username}-changed-123"},
        )
        assert changed.status_code == 200
        token = changed.json()["access_token"]
    return token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_questions_and_knowledge(logger: LearningLogger, tmp_path: Path) -> None:
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    questions = [
        "5G NAS 认证流程有哪些关键风险点？",
        "NAS 鉴权过程中重放风险怎么分析？",
        "基于区块链的隐蔽通信怎么设计实验？",
        "LLM 漏洞检测如何减少误报？",
    ]
    for question in questions:
        logger.log_question(
            "student_demo",
            question,
            {
                "answer": "demo",
                "module": "5G NAS" if "NAS" in question or "认证" in question else "创新专题",
                "intent": "course_qa",
                "confidence": 0.8,
                "abstained": False,
                "course_id": course_id,
                "class_id": class_id,
            },
        )
    document = Document(
        source_path=tmp_path / "nas-faq.md",
        filename="nas-faq.md",
        file_type=".md",
        title="NAS FAQ Evidence",
        text="5G NAS authentication should evaluate replay protection, identity exposure, and security context handling.",
        pages=[
            Page(
                text="5G NAS authentication should evaluate replay protection, identity exposure, and security context handling.",
                page=1,
                section="faq",
                source_type="markdown",
            )
        ],
    )
    saved = logger.save_document(
        document=document,
        content_hash=sha256_bytes(document.text.encode("utf-8")),
        chunks=[document.text],
        byte_size=len(document.text),
        index_status="INDEXED",
        course_id=course_id,
        class_id=class_id,
    )
    logger.set_document_publication(int(saved["id"]), published=True, actor_id=1)


def test_semantic_questions_cluster_and_unrelated_questions_remain_separate(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client)
        headers = _headers(token)
        rebuild = client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=headers,
            json={"distance_threshold": 0.1, "min_cluster_size": 2},
        )
        clusters = client.get("/api/teacher/question-clusters", headers=headers).json()["clusters"]

        assert rebuild.status_code == 200
        nas_clusters = [cluster for cluster in clusters if cluster["label"] == "NAS 安全问题"]
        assert nas_clusters
        assert nas_clusters[0]["count"] == 2
        noise = [cluster for cluster in clusters if cluster["is_noise"]]
        assert noise
        assert all(cluster["count"] == 1 for cluster in noise)
    finally:
        _restore(client)


def test_teacher_can_rename_merge_split_and_approve(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client)
        headers = _headers(token)
        client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=headers,
            json={"distance_threshold": 0.1, "min_cluster_size": 2},
        )
        clusters = client.get("/api/teacher/question-clusters", headers=headers).json()["clusters"]
        target = clusters[0]
        source = clusters[-1]

        renamed = client.patch(
            f"/api/teacher/question-clusters/{target['id']}",
            headers=headers,
            json={"label": "教师重命名 NAS FAQ"},
        )
        merged = client.post(
            "/api/teacher/question-clusters/merge",
            headers=headers,
            json={"target_cluster_id": target["id"], "source_cluster_ids": [source["id"]], "label": "合并后的 FAQ"},
        )
        after_merge = client.get("/api/teacher/question-clusters", headers=headers).json()["clusters"]
        merged_cluster = next(cluster for cluster in after_merge if cluster["id"] == target["id"])
        moved_question_id = merged_cluster["members"][-1]["question_id"]
        split = client.post(
            f"/api/teacher/question-clusters/{target['id']}/split",
            headers=headers,
            json={"question_ids": [moved_question_id], "label": "拆分问题"},
        )
        approved = client.post(f"/api/teacher/question-clusters/{target['id']}/approve", headers=headers)

        assert renamed.status_code == 200
        assert merged.status_code == 200
        assert split.status_code == 200
        assert split.json()["new_cluster_id"] != target["id"]
        assert approved.json()["approved"] is True
    finally:
        _restore(client)


def test_rebuild_keeps_same_course_questions_in_separate_class_clusters(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client)
        headers = _headers(token)
        course_id = main.learning_logger.default_course_id()
        class_a = main.learning_logger.default_class_id()
        with main.learning_logger._connect() as conn:
            teacher = main.learning_logger.user_by_username("teacher_demo")
            student = main.learning_logger.user_by_username("student_demo2")
            cursor = conn.execute(
                """
                INSERT INTO classes (course_id, slug, name, term, teacher_id)
                VALUES (?, 'same-course-cluster-b', 'Same Course Cluster B', '2026', ?)
                """,
                (course_id, teacher["id"]),
            )
            class_b = int(cursor.lastrowid)
            conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_b, teacher["id"]))
            conn.execute("INSERT OR IGNORE INTO class_members (class_id, user_id) VALUES (?, ?)", (class_b, student["id"]))
        for question in [
            "Class B 5G NAS authentication replay risk?",
            "Class B NAS authentication procedure threat?",
        ]:
            main.learning_logger.log_question(
                "student_demo2",
                question,
                {
                    "answer": "demo",
                    "module": "5G NAS",
                    "intent": "course_qa",
                    "confidence": 0.8,
                    "abstained": False,
                    "course_id": course_id,
                    "class_id": class_b,
                },
            )
        response = client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=headers,
            json={"course_id": course_id, "distance_threshold": 0.1, "min_cluster_size": 2},
        )
        clusters = client.get(
            "/api/teacher/question-clusters",
            headers=headers,
            params={"course_id": course_id},
        ).json()["clusters"]
        class_counts = {}
        for cluster in clusters:
            if cluster["is_noise"]:
                continue
            class_counts.setdefault(cluster["class_id"], 0)
            class_counts[cluster["class_id"]] += cluster["count"]
            member_class_ids = {member["class_id"] for member in cluster["members"]}
            assert member_class_ids == {cluster["class_id"]}

        assert response.status_code == 200
        assert class_counts.get(class_a, 0) >= 2
        assert class_counts.get(class_b, 0) >= 2
    finally:
        _restore(client)


def test_migration_13_repairs_existing_mixed_class_clusters_idempotently(tmp_path):
    logger = LearningLogger(tmp_path / "mixed-clusters.db")
    logger.seed_demo_identity()
    course_id = logger.default_course_id()
    class_a = logger.default_class_id()
    with logger._connect() as conn:
        teacher = logger.user_by_username("teacher_demo")
        student = logger.user_by_username("student_demo2")
        cursor = conn.execute(
            """
            INSERT INTO classes (course_id, slug, name, term, teacher_id)
            VALUES (?, 'migration-class-b', 'Migration Class B', '2026', ?)
            """,
            (course_id, teacher["id"]),
        )
        class_b = int(cursor.lastrowid)
        conn.execute("INSERT INTO class_members (class_id, user_id) VALUES (?, ?)", (class_b, teacher["id"]))
        conn.execute("INSERT OR IGNORE INTO class_members (class_id, user_id) VALUES (?, ?)", (class_b, student["id"]))
    qid_a = logger.log_question(
        "student_demo",
        "Class A NAS authentication migration question",
        {
            "answer": "demo",
            "module": "5G NAS",
            "intent": "course_qa",
            "confidence": 0.8,
            "abstained": False,
            "course_id": course_id,
            "class_id": class_a,
        },
    )
    qid_b = logger.log_question(
        "student_demo2",
        "Class B NAS authentication migration question",
        {
            "answer": "demo",
            "module": "5G NAS",
            "intent": "course_qa",
            "confidence": 0.8,
            "abstained": False,
            "course_id": course_id,
            "class_id": class_b,
        },
    )
    cluster_id = logger.create_question_cluster(
        label="Mixed migration cluster",
        representative_question="Mixed migration cluster",
        course_id=course_id,
        class_id=class_a,
        approved=False,
        is_noise=False,
        source="auto",
    )
    logger.add_question_cluster_member(cluster_id=cluster_id, question_id=qid_a, is_representative=True)
    logger.add_question_cluster_member(cluster_id=cluster_id, question_id=qid_b, is_representative=False)
    with logger._connect() as conn:
        before_members = conn.execute("SELECT COUNT(*) FROM question_cluster_members").fetchone()[0]
        conn.execute("DELETE FROM schema_migrations WHERE version = 13")

    logger.init_db()

    with logger._connect() as conn:
        after_members = conn.execute("SELECT COUNT(*) FROM question_cluster_members").fetchone()[0]
        mixed_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT c.id
                FROM question_clusters c
                JOIN question_cluster_members m ON m.cluster_id = c.id
                JOIN questions q ON q.id = m.question_id
                GROUP BY c.id
                HAVING COUNT(DISTINCT q.class_id) > 1
            )
            """
        ).fetchone()[0]
        repair_count = conn.execute("SELECT COUNT(*) FROM question_cluster_repair_log").fetchone()[0]
        first_cluster_count = conn.execute("SELECT COUNT(*) FROM question_clusters").fetchone()[0]

    logger.init_db()
    with logger._connect() as conn:
        second_cluster_count = conn.execute("SELECT COUNT(*) FROM question_clusters").fetchone()[0]
        second_repair_count = conn.execute("SELECT COUNT(*) FROM question_cluster_repair_log").fetchone()[0]

    assert after_members == before_members
    assert mixed_count == 0
    assert repair_count >= 2
    assert second_cluster_count == first_cluster_count
    assert second_repair_count == repair_count


def test_approved_teacher_label_survives_auto_rebuild(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, labeler=MockLabeler("自动标签 A"))
    try:
        token = _login(client)
        headers = _headers(token)
        client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=headers,
            json={"distance_threshold": 0.1, "min_cluster_size": 2},
        )
        cluster = client.get("/api/teacher/question-clusters", headers=headers).json()["clusters"][0]
        client.patch(
            f"/api/teacher/question-clusters/{cluster['id']}",
            headers=headers,
            json={"label": "教师批准标签"},
        )
        client.post(f"/api/teacher/question-clusters/{cluster['id']}/approve", headers=headers)

        main.question_clustering_service_factory = lambda storage: QuestionClusteringService(
            storage,
            embedding=FixedEmbedding(),
            labeler=MockLabeler("自动标签 B"),
        )
        client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=headers,
            json={"distance_threshold": 0.1, "min_cluster_size": 2},
        )
        labels = [item["label"] for item in client.get("/api/teacher/question-clusters", headers=headers).json()["clusters"]]

        assert "教师批准标签" in labels
        assert "自动标签 B" not in labels[:1]
    finally:
        _restore(client)


def test_labeler_failure_falls_back_to_keyword_label(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, labeler=FailingLabeler())
    try:
        token = _login(client)
        headers = _headers(token)
        client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=headers,
            json={"distance_threshold": 0.1, "min_cluster_size": 2},
        )
        clusters = client.get("/api/teacher/question-clusters", headers=headers).json()["clusters"]

        assert any(cluster["label"] in {"5G NAS", "NAS"} for cluster in clusters)
    finally:
        _restore(client)


def test_faq_export_contains_answer_and_citations(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client)
        headers = _headers(token)
        client.post(
            "/api/teacher/question-clusters/rebuild",
            headers=headers,
            json={"distance_threshold": 0.1, "min_cluster_size": 2},
        )
        clusters = client.get("/api/teacher/question-clusters", headers=headers).json()["clusters"]
        cluster = next(item for item in clusters if not item["is_noise"] and item["count"] >= 2)

        draft_export = client.get("/api/teacher/question-clusters/faq-export", headers=headers)
        assert draft_export.status_code == 200
        assert draft_export.json()["items"] == []

        client.post(f"/api/teacher/question-clusters/{cluster['id']}/approve", headers=headers)
        export = client.get("/api/teacher/question-clusters/faq-export", headers=headers)
        item = export.json()["items"][0]

        assert export.status_code == 200
        assert item["label"]
        assert item["representative_question"]
        assert item["count"] >= 2
        assert "[S1]" in item["answer"]
        assert item["citations"]
        assert {"document_id", "chunk_id", "snippet"}.issubset(item["citations"][0])
    finally:
        _restore(client)


def test_faq_export_skips_approved_cluster_without_evidence(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    try:
        token = _login(client)
        headers = _headers(token)
        qid = main.learning_logger.log_question(
            "student_demo",
            "完全没有本地证据支撑的专题问题",
            {
                "answer": "demo",
                "module": "unknown",
                "intent": "course_qa",
                "confidence": 0.5,
                "abstained": False,
                "course_id": main.learning_logger.default_course_id(),
                "class_id": main.learning_logger.default_class_id(),
            },
        )
        cluster_id = main.learning_logger.create_question_cluster(
            label="No Evidence",
            representative_question="完全没有本地证据支撑的专题问题",
            course_id=main.learning_logger.default_course_id(),
            class_id=main.learning_logger.default_class_id(),
            approved=True,
            is_noise=False,
            source="teacher",
        )
        main.learning_logger.add_question_cluster_member(cluster_id=cluster_id, question_id=qid, is_representative=True)

        export = client.get("/api/teacher/question-clusters/faq-export", headers=headers)

        assert export.status_code == 200
        assert all(item["cluster_id"] != cluster_id for item in export.json()["items"])
    finally:
        _restore(client)


def test_faq_export_retrieves_evidence_in_cluster_course_scope(tmp_path):
    logger = LearningLogger(tmp_path / "faq-course-scope.db")
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    qid = logger.log_question(
        "student_demo",
        "5G NAS authentication scoped FAQ",
        {
            "answer": "demo",
            "module": "5G NAS",
            "intent": "course_qa",
            "confidence": 0.9,
            "abstained": False,
            "course_id": course_id,
            "class_id": class_id,
        },
    )
    cluster_id = logger.create_question_cluster(
        label="Scoped FAQ",
        representative_question="5G NAS authentication scoped FAQ",
        course_id=course_id,
        class_id=class_id,
        approved=True,
        is_noise=False,
        source="teacher",
    )
    logger.add_question_cluster_member(cluster_id=cluster_id, question_id=qid, is_representative=True)
    retriever = RecordingFaqRetriever()

    export = QuestionClusteringService(logger, retriever=retriever).export_faq(course_id=course_id)

    assert export["items"]
    assert retriever.course_ids == [course_id]


def test_default_tfidf_clusters_nas_auth_synonyms_without_fixed_embedding(tmp_path):
    logger = LearningLogger(tmp_path / "default-clusters.db")
    _seed_default_cluster_questions(logger)

    service = QuestionClusteringService(logger)
    result = service.rebuild()
    clusters = service.list_clusters()["clusters"]
    nas_clusters = [
        cluster
        for cluster in clusters
        if any("NAS" in member["question"] or "非接入层" in member["question"] for member in cluster["members"])
        and not cluster["is_noise"]
    ]

    assert result["questions_clustered"] == 5
    assert nas_clusters
    assert nas_clusters[0]["count"] == 3
    nas_member_text = " ".join(member["question"] for member in nas_clusters[0]["members"])
    assert "认证" in nas_member_text
    assert "鉴权" in nas_member_text
    assert "非接入层" in nas_member_text
    singleton_questions = [
        member["question"]
        for cluster in clusters
        if cluster["is_noise"]
        for member in cluster["members"]
    ]
    assert any("区块链" in question for question in singleton_questions)
    assert any("漏洞" in question for question in singleton_questions)


def test_default_rebuild_preserves_approved_cluster_label(tmp_path):
    logger = LearningLogger(tmp_path / "approved-default-clusters.db")
    _seed_default_cluster_questions(logger)
    service = QuestionClusteringService(logger)
    service.rebuild()
    nas_cluster = next(
        cluster
        for cluster in service.list_clusters()["clusters"]
        if not cluster["is_noise"] and cluster["count"] >= 2
    )
    service.rename(nas_cluster["id"], "教师审批 NAS 鉴权 FAQ")
    service.approve(nas_cluster["id"])

    result = service.rebuild()
    clusters = service.list_clusters()["clusters"]
    approved = [cluster for cluster in clusters if cluster["approved"]]

    assert result["preserved_questions"] == 3
    assert approved
    assert approved[0]["label"] == "教师审批 NAS 鉴权 FAQ"
    assert approved[0]["count"] == 3


def test_auto_cluster_rebuild_triggers_after_new_question_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("QUESTION_CLUSTER_AUTO_THRESHOLD", "2")
    client = _client(tmp_path, monkeypatch)
    try:
        main._auto_cluster_new_question_counts.clear()
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()

        main._record_question(
            "student_demo",
            "5G NAS auto cluster authentication risk one",
            {"answer": "demo", "module": "5G NAS", "intent": "course_qa", "confidence": 0.9, "abstained": False, "course_id": course_id, "class_id": class_id},
        )
        assert main.learning_logger.question_clusters(course_id=course_id) == []

        main._record_question(
            "student_demo",
            "NAS authentication auto cluster risk two",
            {"answer": "demo", "module": "5G NAS", "intent": "course_qa", "confidence": 0.9, "abstained": False, "course_id": course_id, "class_id": class_id},
        )
        clusters = main.learning_logger.question_clusters(course_id=course_id)

        assert any(not cluster["is_noise"] and cluster["count"] >= 2 for cluster in clusters)
    finally:
        main._auto_cluster_new_question_counts.clear()
        _restore(client)


def test_auto_cluster_rebuild_is_scoped_by_course(tmp_path, monkeypatch):
    monkeypatch.setenv("QUESTION_CLUSTER_AUTO_THRESHOLD", "2")
    client = _client(tmp_path, monkeypatch)
    try:
        main._auto_cluster_new_question_counts.clear()
        course_a = main.learning_logger.default_course_id()
        with main.learning_logger._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
                ("auto-cluster-course-b", "Auto Cluster Course B"),
            )
            course_b = int(cursor.lastrowid)
            class_cursor = conn.execute(
                "INSERT INTO classes (course_id, slug, name, term, teacher_id) VALUES (?, ?, ?, ?, NULL)",
                (course_b, "auto-cluster-class-b", "Auto Cluster Class B", "2026"),
            )
            class_b = int(class_cursor.lastrowid)

        for index in range(2):
            main._record_question(
                "student_demo",
                f"Course A NAS authentication scoped question {index}",
                {"answer": "demo", "module": "5G NAS", "intent": "course_qa", "confidence": 0.9, "abstained": False, "course_id": course_a, "class_id": None},
            )
        for index in range(2):
            main._record_question(
                "student_demo",
                f"Course B NAS authentication scoped question {index}",
                {"answer": "demo", "module": "5G NAS", "intent": "course_qa", "confidence": 0.9, "abstained": False, "course_id": course_b, "class_id": class_b},
            )

        cluster_a_questions = [
            member["question"]
            for cluster in main.learning_logger.question_clusters(course_id=course_a)
            for member in cluster["members"]
        ]
        cluster_b_questions = [
            member["question"]
            for cluster in main.learning_logger.question_clusters(course_id=course_b)
            for member in cluster["members"]
        ]

        assert any("Course A" in question for question in cluster_a_questions)
        assert all("Course B" not in question for question in cluster_a_questions)
        assert any("Course B" in question for question in cluster_b_questions)
        assert all("Course A" not in question for question in cluster_b_questions)
    finally:
        main._auto_cluster_new_question_counts.clear()
        _restore(client)


def test_auto_cluster_rebuild_preserves_approved_faq(tmp_path, monkeypatch):
    monkeypatch.setenv("QUESTION_CLUSTER_AUTO_THRESHOLD", "2")
    client = _client(tmp_path, monkeypatch)
    try:
        main._auto_cluster_new_question_counts.clear()
        course_id = main.learning_logger.default_course_id()
        class_id = main.learning_logger.default_class_id()

        for index in range(2):
            main._record_question(
                "student_demo",
                f"Approved NAS authentication question {index}",
                {"answer": "demo", "module": "5G NAS", "intent": "course_qa", "confidence": 0.9, "abstained": False, "course_id": course_id, "class_id": class_id},
            )
        cluster = next(item for item in main.learning_logger.question_clusters(course_id=course_id) if not item["is_noise"])
        main.learning_logger.rename_question_cluster(cluster["id"], "Teacher approved auto FAQ")
        main.learning_logger.approve_question_cluster(cluster["id"])

        for index in range(2):
            main._record_question(
                "student_demo",
                f"New NAS authentication question after approval {index}",
                {"answer": "demo", "module": "5G NAS", "intent": "course_qa", "confidence": 0.9, "abstained": False, "course_id": course_id, "class_id": class_id},
            )
        clusters = main.learning_logger.question_clusters(course_id=course_id)
        approved = [item for item in clusters if item["approved"]]

        assert approved
        assert approved[0]["label"] == "Teacher approved auto FAQ"
        assert approved[0]["count"] >= 2
    finally:
        main._auto_cluster_new_question_counts.clear()
        _restore(client)


def _seed_default_cluster_questions(logger: LearningLogger) -> None:
    course_id = logger.default_course_id()
    class_id = logger.default_class_id()
    questions = [
        ("5G NAS 认证流程有哪些安全风险？", "5G NAS"),
        ("NAS 鉴权过程里的重放问题怎么分析？", "5G NAS"),
        ("非接入层身份验证有哪些安全隐患？", "5G NAS"),
        ("区块链隐蔽通信实验怎么设计？", "区块链"),
        ("CVE 漏洞样例如何用于课程分析？", "漏洞"),
    ]
    for question, module in questions:
        logger.log_question(
            "student_demo",
            question,
            {
                "answer": "demo",
                "module": module,
                "intent": "course_qa",
                "confidence": 0.8,
                "abstained": False,
                "course_id": course_id,
                "class_id": class_id,
            },
        )
