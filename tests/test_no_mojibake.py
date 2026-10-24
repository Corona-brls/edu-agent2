from pathlib import Path


ROOTS = [Path("backend"), Path("mcp_server"), Path("tests"), Path("frontend")]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}

# Common fragments produced when UTF-8 Chinese text is decoded or re-saved
# with the wrong legacy code page. Keep patterns specific enough to avoid
# rejecting ordinary Chinese prose.
MOJIBAKE_PATTERNS = [
    "\ufffd",
    "锟",
    "銆",
    "鈥",
    "瀹夊",
    "绗",
    "妯″",
    "鍚庣",
    "璇剧",
    "闂",
    "鐭ヨ",
    "鍥寸",
    "閫氫",
    "婕忔",
]


def test_source_files_do_not_contain_common_mojibake():
    offenders: list[str] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name == "test_no_mojibake.py":
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits = [pattern for pattern in MOJIBAKE_PATTERNS if pattern in text]
            if hits:
                offenders.append(f"{path}: {', '.join(hits[:5])}")

    assert offenders == []
