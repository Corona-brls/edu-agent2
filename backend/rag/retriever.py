"""Local TF-IDF retriever for Markdown knowledge chunks."""

from pathlib import Path

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.rag.loader import KnowledgeChunk, load_knowledge_chunks
from backend.core.config import get_settings

STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "或",
    "在",
    "里",
    "中",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "这个",
    "那个",
    "哪些",
    "什么",
    "怎么",
    "如何",
    "问题",
    "是否",
    "有没有",
    "请",
    "帮",
    "帮我",
    "完全",
    "无关",
    "今天",
    "一下",
    "可以",
    "能",
    "能否",
    "一下",
    "？",
    "?",
    "：",
    ":",
    "，",
    ",",
    "。",
    ".",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in jieba.lcut(text):
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isascii():
            continue
        tokens.append(token)
    return tokens


class LocalRetriever:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else get_settings().knowledge_base_dir
        self.chunks: list[KnowledgeChunk] = load_knowledge_chunks(self.base_dir)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
        self.matrix = None
        if self.chunks:
            corpus = [chunk.text for chunk in self.chunks]
            self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not query.strip() or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        tfidf_scores = (self.matrix @ query_vector.T).toarray().ravel()
        query_tokens = set(tokenize(query))
        scores = []
        for index, chunk in enumerate(self.chunks):
            chunk_tokens = set(tokenize(chunk.text))
            overlap = _token_overlap(query_tokens, chunk_tokens)
            heading_boost = _heading_boost(query, chunk.text)
            scores.append(float(tfidf_scores[index]) + overlap * 0.12 + heading_boost)
        scores_array = scores
        ranked_indexes = sorted(range(len(scores_array)), key=scores_array.__getitem__, reverse=True)[:top_k]

        results: list[dict] = []
        for index in ranked_indexes:
            score = float(scores_array[index])
            if score <= 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "source": chunk.source,
                    "title": chunk.title,
                    "snippet": _clean_snippet(chunk.text),
                    "score": round(score, 4),
                    "tfidf_score": round(float(tfidf_scores[index]), 4),
                    "token_overlap": round(_token_overlap(query_tokens, set(tokenize(chunk.text))), 4),
                    "module": chunk.module,
                }
            )
        return results

    def reload(self) -> None:
        self.__init__(self.base_dir)


def _clean_snippet(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(
        line.lstrip("#").strip()
        for line in text.splitlines()
        if line.strip()
    )
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _token_overlap(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def _heading_boost(query: str, text: str) -> float:
    boost = 0.0
    if "风险" in query and "风险" in text:
        boost += 0.08
    if "实验" in query and "实验" in text:
        boost += 0.06
    if "CVE" in query.upper() and "CVE" in text.upper():
        boost += 0.08
    if "3GPP" in query.upper() and "3GPP" in text.upper():
        boost += 0.08
    return boost
