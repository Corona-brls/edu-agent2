"""Load and chunk local Markdown/TXT knowledge files."""

from dataclasses import dataclass
from pathlib import Path

from backend.core.config import get_settings


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    title: str
    text: str
    module: str


def list_knowledge_files(base_dir: str | Path | None = None) -> list[Path]:
    root = Path(base_dir) if base_dir is not None else get_settings().knowledge_base_dir
    if not root.exists():
        return []
    return sorted([*root.glob("*.md"), *root.glob("*.txt")])


def load_knowledge_chunks(base_dir: str | Path | None = None) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for path in list_knowledge_files(base_dir):
        text = path.read_text(encoding="utf-8")
        title = _extract_title(text, path.stem)
        module = _infer_module(path.name)
        for chunk_text in _split_markdown(text):
            chunks.append(
                KnowledgeChunk(
                    source=path.name,
                    title=title,
                    text=chunk_text,
                    module=module,
                )
            )
    return chunks


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return fallback


def _split_markdown(text: str, max_chars: int = 700) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = [stripped]
            continue
        if stripped:
            current.append(stripped)

    if current:
        blocks.append("\n".join(current).strip())

    chunks: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        paragraphs = [part.strip() for part in block.split("\n") if part.strip()]
        current_text = ""
        for paragraph in paragraphs:
            candidate = f"{current_text}\n{paragraph}".strip()
            if len(candidate) > max_chars and current_text:
                chunks.append(current_text)
                current_text = paragraph
            else:
                current_text = candidate
        if current_text:
            chunks.append(current_text)
    return chunks


def _infer_module(filename: str) -> str:
    mapping = {
        "course_intro.md": "课程智能体",
        "5g_nas.md": "5G NAS",
        "blockchain_covert.md": "区块链隐蔽通信",
        "llm_vuln_detection.md": "LLM 漏洞检测",
        "cve_samples.md": "漏洞样例",
        "standards_samples.md": "标准文档",
    }
    return mapping.get(filename, "课程资料")
