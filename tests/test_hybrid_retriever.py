from pathlib import Path

import pytest

from backend.rag.document_parser import Document, Page, chunk_document_text, parse_document
from backend.rag.hybrid_retriever import (
    classify_course_query_update,
    CourseDocumentConstraintError,
    HybridRetriever,
    _keyword_scores,
    _query_coverage,
    parse_explicit_course_document_title,
    query_fact_fingerprint,
    retrieval_snippet,
    resolve_course_document_constraint,
)
from backend.rag.vector_store import ChromaVectorStore
from backend.storage.logger import LearningLogger, sha256_text


class EvalEmbedding:
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
            float(lowered.count("authentication")),
            float(lowered.count("pdf")),
            float(lowered.count("docx")),
            float(lowered.count("standard")),
            float(lowered.count("blockchain")),
        ]


class BrokenVectorStore:
    def query(self, text: str, top_k: int = 5):
        raise RuntimeError("chroma unavailable")


class RankedVectorStore:
    def __init__(self, matches: list[dict]):
        self.matches = matches
        self.requests: list[int] = []

    def query(self, text: str, top_k: int = 5):
        self.requests.append(top_k)
        return self.matches[:top_k]


class MutatingVectorStore:
    def __init__(self, match: dict, mutate):
        self.match = match
        self.mutate = mutate
        self.calls = 0

    def query(self, text: str, top_k: int = 5):
        self.calls += 1
        self.mutate()
        return [self.match]


def _save_published_document(storage, *, course_id, filename, title, chunks):
    text = "\n".join(chunks)
    document = Document(
        source_path=Path("knowledge_base") / filename,
        filename=filename,
        file_type="md",
        title=title,
        text=text,
        pages=[Page(text=text, page=1, section=title, source_type="markdown")],
    )
    saved = storage.save_document(
        document=document,
        content_hash=sha256_text(text),
        chunks=chunks,
        byte_size=len(text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)
    return document_id, [int(item["id"]) for item in storage.chunks_for_document(document_id)]


def test_explicit_document_filter_applies_before_vector_scoring_and_keeps_answer_chunk(tmp_path):
    storage = LearningLogger(tmp_path / "explicit-title-vector.db")
    course_id = storage.default_course_id()
    target_id, target_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="target.md",
        title="目标教材",
        chunks=[
            "目标教材 导语和目录，不包含问题答案。",
            "TARGET_FIELD=目标教材中的正确回答字段。",
        ],
    )
    distractor_id, distractor_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="distractor.md",
        title="相似资料",
        chunks=["TARGET_FIELD TARGET_FIELD 目标教材 正确回答字段 干扰内容。"],
    )
    vector_store = RankedVectorStore(
        [
            {"id": str(distractor_chunks[0]), "metadata": {"chunk_id": distractor_chunks[0]}, "score": 0.99},
            {"id": str(target_chunks[1]), "metadata": {"chunk_id": target_chunks[1]}, "score": 0.90},
            {"id": str(target_chunks[0]), "metadata": {"chunk_id": target_chunks[0]}, "score": 0.80},
        ]
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=vector_store,
        min_score=0.0,
    ).retrieve(
        "请只依据当前课程发布的《目标教材》说明 TARGET_FIELD",
        top_k=2,
        course_id=course_id,
        allowed_document_ids={target_id},
    )

    assert response.abstained is False
    assert {item.document_id for item in response.results} == {target_id}
    assert target_chunks[1] in {item.chunk_id for item in response.results}
    assert distractor_id not in {item.document_id for item in response.results}


def test_explicit_document_filter_applies_to_keyword_fallback(tmp_path):
    storage = LearningLogger(tmp_path / "explicit-title-keyword.db")
    course_id = storage.default_course_id()
    target_id, target_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="target.md",
        title="目标教材",
        chunks=["目标教材目录", "KEYWORD_TARGET 目标教材唯一答案"],
    )
    distractor_id, _ = _save_published_document(
        storage,
        course_id=course_id,
        filename="distractor.md",
        title="干扰教材",
        chunks=["KEYWORD_TARGET KEYWORD_TARGET KEYWORD_TARGET 干扰答案"],
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=BrokenVectorStore(),
        min_score=0.0,
    ).retrieve(
        "请只根据《目标教材》解释 KEYWORD_TARGET",
        top_k=2,
        course_id=course_id,
        allowed_document_ids={target_id},
    )

    assert response.fallback_used is True
    assert {item.document_id for item in response.results} == {target_id}
    assert target_chunks[1] in {item.chunk_id for item in response.results}
    assert distractor_id not in {item.document_id for item in response.results}


def test_explicit_document_keyword_fallback_keeps_bounded_low_score_support(tmp_path):
    storage = LearningLogger(tmp_path / "explicit-title-bounded-support.db")
    course_id = storage.default_course_id()
    query = (
        "请只依据当前课程发布的《目标教材》回答：在合成消息序列的防御性实验中，"
        "应记录哪些字段？请给出课程资料引用，不要提供真实网络攻击步骤。"
    )
    anchor = (
        "# 目标教材\nsource_id target review_state approved 目标教材 当前课程发布 "
        "课程资料 引用 防御性实验 合成消息序列"
    ) * 5
    support = (
        "在隔离的合成消息序列防御性实验中，每条记录应保存输入编号、初始状态、"
        "预期转移、实际转移、拒绝原因和处理时延。"
        + " 补充说明用于离线教学分析，不包含攻击步骤。" * 12
    )
    noise = "天气烹饪诗歌与遥远主题完全无关。" * 20
    target_id, target_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="target-long.md",
        title="目标教材",
        chunks=[anchor, support, noise],
    )
    distractor_id, _ = _save_published_document(
        storage,
        course_id=course_id,
        filename="distractor-long.md",
        title="高相似竞争资料",
        chunks=[query * 8],
    )
    target_rows = [
        chunk
        for chunk in storage.document_chunks(course_id=course_id)
        if int(chunk["document_id"]) == target_id
    ]
    keyword_scores = _keyword_scores(query, target_rows, top_k=len(target_rows))

    assert keyword_scores[target_chunks[0]] >= 0.18
    assert 0 < keyword_scores[target_chunks[1]] < 0.18
    assert _query_coverage(query, support) >= 0.2
    assert keyword_scores.get(target_chunks[2], 0.0) == 0.0

    response = HybridRetriever(
        storage=storage,
        vector_store=BrokenVectorStore(),
        min_score=0.18,
    ).retrieve(
        query,
        top_k=2,
        course_id=course_id,
        allowed_document_ids={target_id},
    )

    assert response.fallback_used is True
    assert response.abstained is False
    assert [item.chunk_id for item in response.results] == target_chunks[:2]
    assert all(item.document_id == target_id for item in response.results)
    assert distractor_id not in {item.document_id for item in response.results}
    assert target_chunks[2] not in {item.chunk_id for item in response.results}
    assert len(response.results) <= 2


def test_explicit_document_healthy_vector_keeps_support_once_and_within_top_k(tmp_path):
    storage = LearningLogger(tmp_path / "explicit-title-vector-support.db")
    course_id = storage.default_course_id()
    query = "目标教材 合成消息序列 防御性实验 记录字段"
    anchor = "目标教材 当前课程资料 合成消息序列 防御性实验 记录字段 " * 8
    support = (
        "合成消息序列防御性实验的记录字段如下：每条记录应保存输入编号、初始状态、"
        "预期转移、实际转移、拒绝原因和处理时延。"
        + " 仅用于隔离教学分析。" * 18
    )
    noise = "天气烹饪诗歌与遥远主题完全无关。" * 20
    target_id, target_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="target-vector-support.md",
        title="目标教材",
        chunks=[anchor, support, noise],
    )
    vector_store = RankedVectorStore(
        [
            {"id": str(target_chunks[0]), "metadata": {"chunk_id": target_chunks[0]}, "score": 0.9},
            {"id": str(target_chunks[1]), "metadata": {"chunk_id": target_chunks[1]}, "score": 0.01},
        ]
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=vector_store,
        min_score=0.18,
    ).retrieve(
        query,
        top_k=2,
        course_id=course_id,
        allowed_document_ids={target_id},
    )

    assert response.fallback_used is False
    assert len(response.results) == 2
    assert len({item.chunk_id for item in response.results}) == 2
    assert set(item.chunk_id for item in response.results) == set(target_chunks[:2])
    assert target_chunks[2] not in {item.chunk_id for item in response.results}


def test_explicit_document_snippet_selects_complete_late_support_sentences(tmp_path):
    storage = LearningLogger(tmp_path / "explicit-title-late-support.db")
    course_id = storage.default_course_id()
    query = (
        "请只依据《网络安全实验设计与评价方法卡》说明基线组、处理组、"
        "固定变量和重复实验如何设计，并给出评价指标与复现实验记录。"
    )
    metadata_prefix = (
        "教材标题与审阅元数据；source_id: shared.experiment_design.v1；"
        "review_state: approved；本段仅说明目录结构和适用边界。"
        + "课程目录占位说明，不是问题答案。" * 12
    )
    late_support = (
        "基线组保持默认处理，处理组启用待比较策略。"
        "固定变量包括同一批隔离日志、相同随机种子与相同运行轮次。"
        "重复实验至少执行三轮并记录每轮差异。"
    )
    adjacent_support = "评价指标采用误报率与漏报率。复现实验记录应保存参数、版本和结果摘要。"
    target_id, target_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="experiment-design.md",
        title="网络安全实验设计与评价方法卡",
        chunks=[metadata_prefix + late_support, adjacent_support],
    )
    distractor_id, _ = _save_published_document(
        storage,
        course_id=course_id,
        filename="distractor.md",
        title="实验设计相似资料",
        chunks=[query * 6],
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=BrokenVectorStore(),
        min_score=0.0,
    ).retrieve(
        query,
        top_k=2,
        course_id=course_id,
        allowed_document_ids={target_id},
    )

    assert response.abstained is False
    assert {item.document_id for item in response.results} == {target_id}
    assert distractor_id not in {item.document_id for item in response.results}
    snippets = {item.chunk_id: item.snippet for item in response.results}
    assert "基线组保持默认处理，处理组启用待比较策略。" in snippets[target_chunks[0]]
    assert "固定变量包括同一批隔离日志、相同随机种子与相同运行轮次。" in snippets[target_chunks[0]]
    assert "重复实验至少执行三轮并记录每轮差异。" in snippets[target_chunks[0]]
    assert "评价指标采用误报率与漏报率。" in snippets[target_chunks[1]]
    assert all(len(value) <= 240 for value in snippets.values())


KB9_EXACT_QUESTION = (
    "请只根据《网络安全实验设计与评价方法卡》解释基线、对照、评价指标和复现实验记录的关系。"
)


def test_explicit_textbook_fact_fingerprint_ignores_control_and_title_but_detects_new_facts():
    original = "请只根据《目标教材》解释基线与对照的关系。"
    control_only = (
        original
        + "\n补充信息: 请继续根据《目标教材》回答，不增加新的事实主题。"
    )
    with_new_fact = original + "\n补充信息: 再说明误报率和样本量。"
    unsupported_new_fact = original + "\n补充信息: 再说明量子纠错预算。"

    assert query_fact_fingerprint(original) == query_fact_fingerprint(control_only)
    assert query_fact_fingerprint(original) != query_fact_fingerprint(with_new_fact)
    assert query_fact_fingerprint(original) != query_fact_fingerprint(unsupported_new_fact)
    assert query_fact_fingerprint(with_new_fact) != query_fact_fingerprint(unsupported_new_fact)


@pytest.mark.parametrize(
    "fact_phrase",
    [
        "实验",
        "记录",
        "评价",
        "方法",
        "设计",
        "实验记录",
        "评价方法",
        "实验设计",
        "记录方法",
        "experiment log",
        "evaluation method",
    ],
)
def test_explicit_textbook_fact_fingerprint_preserves_general_content_terms(fact_phrase):
    assert query_fact_fingerprint(fact_phrase)


@pytest.mark.parametrize(
    "style_only",
    [
        "请详细一点",
        "请展开说明",
        "换一种说法",
        "请简要总结",
        "说明得更具体一些",
        "please continue",
        "please elaborate",
        "请继续根据《网络安全实验设计与评价方法卡》回答。",
    ],
)
def test_explicit_textbook_fact_fingerprint_ignores_style_only_updates(style_only):
    assert query_fact_fingerprint(style_only) == ()


@pytest.mark.parametrize(
    "mixed",
    [
        "请详细说明评价方法",
        "请用另一种说法解释实验记录",
        "please elaborate on the evaluation method",
    ],
)
def test_explicit_textbook_fact_fingerprint_keeps_facts_inside_style_requests(mixed):
    assert query_fact_fingerprint(mixed)


def test_explicit_textbook_query_update_classifies_control_additive_and_replacement():
    previous = "请只根据《目标教材》解释基线与对照的关系。"

    control = classify_course_query_update(previous, "请详细一点")
    additive = classify_course_query_update(previous, "请详细说明评价方法")
    replacement = classify_course_query_update(
        previous,
        "不要再回答基线与对照，只说明误报率和样本量",
    )

    assert control.update_type == "control_only"
    assert control.effective_query == previous
    assert control.required_fact_fingerprint == ()
    assert additive.update_type == "additive"
    assert additive.required_fact_fingerprint
    assert set(additive.previous_fact_fingerprint) < set(additive.effective_fact_fingerprint)
    assert replacement.update_type == "replacement"
    assert replacement.required_fact_fingerprint == replacement.effective_fact_fingerprint
    assert set(replacement.effective_fact_fingerprint).isdisjoint(
        replacement.previous_fact_fingerprint
    )


def test_explicit_textbook_query_update_is_stable_across_sequential_updates():
    previous = "请只根据《目标教材》解释基线与对照的关系。"

    first_control = classify_course_query_update(previous, "请详细一点")
    second_control = classify_course_query_update(first_control.effective_query, "换一种说法")
    additive = classify_course_query_update(second_control.effective_query, "再说明实验记录")
    replacement = classify_course_query_update(
        additive.effective_query,
        "改为只说明评价方法",
    )
    ordinary_negation = classify_course_query_update(
        previous,
        "不要省略实验记录中的异常说明",
    )

    assert first_control.update_type == second_control.update_type == "control_only"
    assert second_control.effective_query == previous
    assert additive.update_type == "additive"
    assert set(additive.previous_fact_fingerprint) < set(additive.effective_fact_fingerprint)
    assert replacement.update_type == "replacement"
    assert set(replacement.effective_fact_fingerprint).isdisjoint(
        additive.previous_fact_fingerprint
    )
    assert ordinary_negation.update_type == "additive"


def test_explicit_kb9_exact_question_selects_method_metrics_and_reproduction(tmp_path):
    storage = LearningLogger(tmp_path / "kb9-exact-question.db")
    course_id = storage.default_course_id()
    front_matter = (
        "# 网络安全实验设计与评价方法卡\n"
        "> source_id: shared.experiment_design.v1\n"
        "> review_state: approved\n"
        "## 方法设计\n"
    )
    method_and_results = (
        "建立基线组与处理组，只改变预先声明的独立变量。"
        "固定软件版本、数据划分、随机种子和运行资源；至少重复运行三次并报告波动。"
        "## 结果分析\n"
        "分类任务至少报告混淆矩阵、精确率、召回率和样本量。"
    )
    adjacent = "复现实验记录应保存版本、数据划分、随机种子、每轮指标和异常说明。"
    target_id, target_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="experiment-design.md",
        title="网络安全实验设计与评价方法卡",
        chunks=[front_matter + ("目录与审阅说明。" * 30) + method_and_results, adjacent],
    )
    distractor_id, _ = _save_published_document(
        storage,
        course_id=course_id,
        filename="distractor.md",
        title="实验设计术语汇总",
        chunks=[KB9_EXACT_QUESTION * 8],
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=BrokenVectorStore(),
        min_score=0.0,
    ).retrieve(
        KB9_EXACT_QUESTION,
        top_k=2,
        course_id=course_id,
        allowed_document_ids={target_id},
        snippet_query=KB9_EXACT_QUESTION,
    )

    assert response.abstained is False
    assert {item.document_id for item in response.results} == {target_id}
    assert distractor_id not in {item.document_id for item in response.results}
    rendered = "".join(item.snippet for item in response.results)
    assert "建立基线组与处理组" in rendered
    assert "固定软件版本、数据划分、随机种子" in rendered
    assert "至少重复运行三次" in rendered
    assert "混淆矩阵、精确率、召回率和样本量" in rendered
    assert target_chunks[0] in {item.chunk_id for item in response.results}
    assert all(len(item.snippet) <= 240 for item in response.results)


def test_explicit_snippet_preserves_markdown_and_english_sentence_boundaries():
    query = "baseline control evaluation metrics reproducible record"
    lf = (
        "# Experiment Design\n"
        "- Baseline uses version 3.2 and dataset split 0.8/0.2\n"
        "- Control changes one declared variable\n"
        "Evaluation metrics include precision and recall. Reproducible records keep https://example.org/v1.2."
    )
    crlf = lf.replace("\n", "\r\n")

    lf_snippet = retrieval_snippet(lf, query=query)
    crlf_snippet = retrieval_snippet(crlf, query=query)

    assert lf_snippet == crlf_snippet
    assert "Baseline uses version 3.2 and dataset split 0.8/0.2" in lf_snippet
    assert "Control changes one declared variable" in lf_snippet
    assert "Evaluation metrics include precision and recall." in lf_snippet
    assert "https://example.org/v1.2" in lf_snippet
    assert len(lf_snippet) <= 240


def test_explicit_snippet_uses_late_bounded_window_instead_of_unrelated_head():
    head = "无关的审阅说明与目录信息 " * 30
    late = (
        "本段很长且直到后部才说明评价指标包括混淆矩阵、精确率、召回率和样本量，"
        "复现实验记录保存版本、数据划分、随机种子和每轮波动。"
    )
    snippet = retrieval_snippet(
        head + late,
        query="评价指标 复现实验记录",
        max_chars=240,
    )

    assert "混淆矩阵" in snippet
    assert "复现实验记录" in snippet
    assert not snippet.startswith("无关的审阅说明")
    assert len(snippet) <= 240


def test_real_kb9_source_parser_chunker_and_hybrid_cover_exact_manual_question(tmp_path):
    source = Path("knowledge_sources/shared/experiment_design_method_card_v1.md")
    document = parse_document(source)
    chunks = chunk_document_text(document.text)
    storage = LearningLogger(tmp_path / "kb9-real-source.db")
    course_id = storage.default_course_id()
    saved = storage.save_document(
        document=document,
        content_hash=sha256_text(document.text),
        chunks=chunks,
        byte_size=source.stat().st_size,
        index_status="INDEXED",
        course_id=course_id,
    )
    document_id = int(saved["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)
    title, allowed = resolve_course_document_constraint(
        storage,
        course_id=course_id,
        user_goal=KB9_EXACT_QUESTION,
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=BrokenVectorStore(),
        min_score=0.0,
    ).retrieve(
        KB9_EXACT_QUESTION,
        top_k=5,
        course_id=course_id,
        allowed_document_ids=allowed,
        snippet_query=KB9_EXACT_QUESTION,
    )

    assert title == "网络安全实验设计与评价方法卡"
    assert allowed == {document_id}
    rendered = "".join(item.snippet for item in response.results)
    assert "建立基线组与处理组" in rendered
    assert "固定软件版本、数据划分、随机种子" in rendered
    assert "至少重复运行三次" in rendered
    assert "混淆矩阵、精确率、召回率和样本量" in rendered
    assert all(item.document_id == document_id for item in response.results)


def test_open_retrieval_keeps_head_snippet_for_late_matching_text(tmp_path):
    storage = LearningLogger(tmp_path / "open-late-support.db")
    course_id = storage.default_course_id()
    prefix = "普通课程摘要头部。" * 40
    late_fact = "基线组和处理组的关键事实位于正文后部。"
    _document_id, chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="open.md",
        title="普通课程资料",
        chunks=[prefix + late_fact],
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=RankedVectorStore(
            [{"id": str(chunks[0]), "metadata": {"chunk_id": chunks[0]}, "score": 0.9}]
        ),
        min_score=0.0,
    ).retrieve("基线组 处理组", top_k=1, course_id=course_id)

    assert response.abstained is False
    assert response.results[0].snippet.startswith("普通课程摘要头部")
    assert late_fact not in response.results[0].snippet


def test_low_score_support_does_not_bypass_threshold_without_unique_constraint(tmp_path):
    storage = LearningLogger(tmp_path / "open-course-threshold.db")
    course_id = storage.default_course_id()
    query = (
        "请只依据当前课程发布的《目标教材》回答：在合成消息序列的防御性实验中，"
        "应记录哪些字段？请给出课程资料引用，不要提供真实网络攻击步骤。"
    )
    anchor = (
        "# 目标教材\nsource_id target review_state approved 目标教材 当前课程发布 "
        "课程资料 引用 防御性实验 合成消息序列"
    ) * 5
    support = (
        "在隔离的合成消息序列防御性实验中，每条记录应保存输入编号、初始状态、"
        "预期转移、实际转移、拒绝原因和处理时延。"
        + " 补充说明用于离线教学分析，不包含攻击步骤。" * 12
    )
    target_id, target_chunks = _save_published_document(
        storage,
        course_id=course_id,
        filename="open-target.md",
        title="目标教材",
        chunks=[anchor, support],
    )

    response = HybridRetriever(
        storage=storage,
        vector_store=BrokenVectorStore(),
        min_score=0.18,
    ).retrieve(query, top_k=2, course_id=course_id)

    assert response.fallback_used is True
    assert response.abstained is False
    assert target_chunks[0] in {item.chunk_id for item in response.results}
    assert target_chunks[1] not in {item.chunk_id for item in response.results}
    assert all(item.document_id == target_id for item in response.results)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("请只依据《目标教材》回答", "目标教材"),
        ("请只根据当前课程发布的《目标　教材》回答", "目标 教材"),
        ("普通问题提到《目标教材》但没有强约束", None),
        ("讨论目标教材这个普通术语", None),
    ],
)
def test_explicit_course_title_parser_is_exact_and_bounded(query, expected):
    assert parse_explicit_course_document_title(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "请只根据《》回答",
        "请只根据《未闭合标题回答",
        "请只根据《教材甲》和《教材乙》回答",
        "请只根据《" + ("超" * 161) + "》回答",
    ],
)
def test_explicit_course_title_parser_rejects_malformed_constraints(query):
    with pytest.raises(CourseDocumentConstraintError) as exc_info:
        parse_explicit_course_document_title(query)
    assert exc_info.value.error_code == "COURSE_DOCUMENT_CONSTRAINT_INVALID"


def test_explicit_course_title_resolution_uses_unique_visible_exact_title(tmp_path):
    storage = LearningLogger(tmp_path / "explicit-title-resolution.db")
    course_id = storage.default_course_id()
    visible_id, _ = _save_published_document(
        storage,
        course_id=course_id,
        filename="visible.md",
        title="目标　教材",
        chunks=["visible"],
    )
    hidden_id, _ = _save_published_document(
        storage,
        course_id=course_id,
        filename="hidden.md",
        title="目标 教材",
        chunks=["hidden"],
    )
    storage.set_document_publication(hidden_id, published=False, actor_id=1)

    title, allowed = resolve_course_document_constraint(
        storage,
        course_id=course_id,
        user_goal="请只依据《目标 教材》回答",
    )

    assert title == "目标 教材"
    assert allowed == frozenset({visible_id})

    storage.set_document_publication(hidden_id, published=True, actor_id=1)
    with pytest.raises(CourseDocumentConstraintError) as exc_info:
        resolve_course_document_constraint(
            storage,
            course_id=course_id,
            user_goal="请只依据《目标 教材》回答",
        )
    assert exc_info.value.error_code == "COURSE_DOCUMENT_AMBIGUOUS"


def test_explicit_course_title_resolution_does_not_cross_course_or_fuzzy_match(tmp_path):
    storage = LearningLogger(tmp_path / "explicit-title-course-scope.db")
    current_course_id = storage.default_course_id()
    with storage._connect() as conn:
        other_course_id = int(
            conn.execute(
                "INSERT INTO courses (slug, name, description, assessment_method) VALUES (?, ?, '', '')",
                ("other-course", "Other Course"),
            ).lastrowid
        )
    _save_published_document(
        storage,
        course_id=current_course_id,
        filename="similar.md",
        title="目标教材补充",
        chunks=["similar"],
    )
    _save_published_document(
        storage,
        course_id=other_course_id,
        filename="other.md",
        title="目标教材",
        chunks=["other course"],
    )

    with pytest.raises(CourseDocumentConstraintError) as exc_info:
        resolve_course_document_constraint(
            storage,
            course_id=current_course_id,
            user_goal="请只根据《目标教材》回答",
        )
    assert exc_info.value.error_code == "COURSE_DOCUMENT_NOT_FOUND"


def test_search_with_metadata_rechecks_sqlite_after_publication_is_withdrawn(tmp_path):
    storage = LearningLogger(tmp_path / "return-recheck.db")
    course_id = storage.default_course_id()
    parsed = Document(
        source_path=Path("course/return-recheck.md"),
        filename="return-recheck.md",
        file_type="md",
        title="Return Recheck",
        text="RETURN_RECHECK_MARKER current published evidence",
        pages=[
            Page(
                text="RETURN_RECHECK_MARKER current published evidence",
                page=1,
                section="recheck",
                source_type="markdown",
            )
        ],
    )
    saved = storage.save_document(
        document=parsed,
        content_hash=sha256_text(parsed.text),
        chunks=[parsed.text],
        byte_size=len(parsed.text.encode("utf-8")),
        index_status="INDEXED",
        course_id=course_id,
    )
    document_id = int(saved["id"])
    chunk_id = int(storage.chunks_for_document(document_id)[0]["id"])
    storage.set_document_publication(document_id, published=True, actor_id=1)
    vector_store = MutatingVectorStore(
        {
            "id": str(chunk_id),
            "metadata": {"chunk_id": chunk_id, "document_id": document_id},
            "score": 0.99,
        },
        lambda: storage.set_document_publication(document_id, published=False, actor_id=1),
    )

    results, fallback_used = HybridRetriever(
        storage=storage,
        vector_store=vector_store,
        min_score=0.01,
    ).search_with_metadata("RETURN_RECHECK_MARKER", top_k=1, course_id=course_id)

    assert vector_store.calls == 1
    assert storage.document(document_id)["published_to_students"] is False
    assert results == []
    assert fallback_used is False


def test_vector_scores_exhaust_candidates_beyond_512_before_stopping(tmp_path):
    legal_chunk_id = 9001
    matches = [
        {
            "id": str(index),
            "metadata": {"chunk_id": index},
            "score": 1.0 - (index / 10_000),
        }
        for index in range(1, 513)
    ]
    matches.append(
        {
            "id": str(legal_chunk_id),
            "metadata": {"chunk_id": legal_chunk_id},
            "score": 0.8,
        }
    )
    vector_store = RankedVectorStore(matches)
    retriever = HybridRetriever(
        storage=LearningLogger(tmp_path / "candidate-exhaustion.db"),
        vector_store=vector_store,
    )

    scores, fallback_used = retriever._vector_scores(
        "vector-only-query",
        [{"id": legal_chunk_id}],
        top_k=1,
    )

    assert scores == {legal_chunk_id: 0.8}
    assert fallback_used is False
    assert max(vector_store.requests) >= 513


def test_hybrid_retriever_fuses_keyword_and_vector_scores(tmp_path):
    storage, vector_store = _build_eval_index(tmp_path)
    retriever = HybridRetriever(storage=storage, vector_store=vector_store, min_score=0.05)

    results = retriever.search("NAS authentication risk", top_k=3)

    assert results
    assert results[0].source == "nas.pdf"
    assert results[0].source_type == "pdf"
    assert results[0].page == 2
    assert results[0].section == "NAS Security"
    assert results[0].keyword_score > 0
    assert results[0].vector_score > 0
    assert results[0].final_score > 0
    assert results[0].chunk_id
    assert results[0].document_id


def test_hybrid_retriever_retrieves_docx_knowledge(tmp_path):
    storage, vector_store = _build_eval_index(tmp_path)
    retriever = HybridRetriever(storage=storage, vector_store=vector_store, min_score=0.05)

    response = retriever.retrieve("DOCX standard experiment", top_k=3)

    assert response.abstained is False
    assert response.results[0].source == "standard.docx"
    assert response.results[0].source_type == "docx"
    assert response.results[0].section == "Standard Experiment"


def test_hybrid_retriever_ignores_non_indexed_documents(tmp_path):
    storage, vector_store = _build_eval_index(tmp_path)
    pending_doc = Document(
        source_path=Path("knowledge_base/pending.pdf"),
        filename="pending.pdf",
        file_type="pdf",
        title="Pending NAS",
        text="NAS authentication pending document should not appear",
        pages=[
            Page(
                text="NAS authentication pending document should not appear",
                page=1,
                section="Pending",
                source_type="pdf",
            )
        ],
    )
    storage.save_document(
        document=pending_doc,
        content_hash=sha256_text(pending_doc.text),
        chunks=None,
        byte_size=len(pending_doc.text.encode("utf-8")),
        index_status="PENDING",
        course_id=storage.default_course_id(),
    )
    retriever = HybridRetriever(storage=storage, vector_store=vector_store, min_score=0.05)

    results = retriever.search("pending document", top_k=5)

    assert all(result.source != "pending.pdf" for result in results)


def test_hybrid_retriever_falls_back_to_keyword_when_chroma_fails(tmp_path):
    storage, _ = _build_eval_index(tmp_path)
    retriever = HybridRetriever(storage=storage, vector_store=BrokenVectorStore(), min_score=0.05)

    response = retriever.retrieve("NAS authentication", top_k=2)

    assert response.fallback_used is True
    assert response.abstained is False
    assert response.results[0].source == "nas.pdf"
    assert response.results[0].metadata["fallback"] is True
    assert response.results[0].vector_score == 0.0


def test_hybrid_retriever_abstains_for_unrelated_question(tmp_path):
    storage, vector_store = _build_eval_index(tmp_path)
    retriever = HybridRetriever(storage=storage, vector_store=vector_store, min_score=0.5)

    response = retriever.retrieve("weather forecast unrelated", top_k=3)

    assert response.abstained is True
    assert response.results == []


def test_hybrid_retriever_keeps_short_nas_query_but_rejects_low_coverage_query(tmp_path):
    storage, vector_store = _build_eval_index(tmp_path)
    retriever = HybridRetriever(storage=storage, vector_store=vector_store, min_score=0.18)

    nas_response = retriever.retrieve("NAS", top_k=3)
    noisy_response = retriever.retrieve(
        "NAS alien weather cooking poetry theorem conclusion unrelated conditions",
        top_k=3,
    )

    assert nas_response.abstained is False
    assert nas_response.results[0].source == "nas.pdf"
    assert noisy_response.abstained is True
    assert noisy_response.results == []


def test_hybrid_retriever_rejects_nonexistent_paper_conclusion(tmp_path):
    storage, vector_store = _build_eval_index(tmp_path)
    retriever = HybridRetriever(storage=storage, vector_store=vector_store, min_score=0.18)

    response = retriever.retrieve("nonexistent paper claims quantum NAS invisibility theorem", top_k=3)

    assert response.abstained is True
    assert response.results == []


def _build_eval_index(tmp_path):
    storage = LearningLogger(tmp_path / "test.db")
    storage.seed_demo_identity()
    course_id = storage.default_course_id()
    documents = [
        Document(
            source_path=Path("knowledge_base/nas.pdf"),
            filename="nas.pdf",
            file_type="pdf",
            title="NAS Security",
            text="PDF NAS authentication risk replay identity protection",
            pages=[
                Page(
                    text="PDF NAS authentication risk replay identity protection",
                    page=2,
                    section="NAS Security",
                    source_type="pdf",
                )
            ],
        ),
        Document(
            source_path=Path("knowledge_base/standard.docx"),
            filename="standard.docx",
            file_type="docx",
            title="Standard Experiment",
            text="DOCX standard experiment procedure for protocol analysis",
            pages=[
                Page(
                    text="DOCX standard experiment procedure for protocol analysis",
                    page=None,
                    section="Standard Experiment",
                    source_type="docx",
                )
            ],
        ),
        Document(
            source_path=Path("knowledge_base/blockchain.md"),
            filename="blockchain.md",
            file_type="md",
            title="Blockchain",
            text="Blockchain covert channel timing and transaction encoding",
            pages=[
                Page(
                    text="Blockchain covert channel timing and transaction encoding",
                    page=1,
                    section="Blockchain",
                    source_type="md",
                )
            ],
        ),
    ]
    document_ids = []
    for document in documents:
        saved = storage.save_document(
            document=document,
            content_hash=sha256_text(document.text),
            chunks=None,
            byte_size=len(document.text.encode("utf-8")),
            index_status="INDEXED",
            course_id=course_id,
        )
        document_ids.append(int(saved["id"]))

    vector_store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="hybrid_eval",
        embedding_function=EvalEmbedding(),
    )
    vector_store.add_chunks(storage.document_chunks(course_id=course_id))
    for document_id in document_ids:
        storage.set_document_publication(document_id, published=True, actor_id=1)
    return storage, vector_store
