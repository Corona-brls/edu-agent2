from __future__ import annotations

import ast
from pathlib import Path


def test_teacher_upload_widget_limits_single_file_to_25_mb() -> None:
    source = Path("frontend/pages/teacher.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    file_uploaders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "file_uploader"
    ]

    assert file_uploaders, "teacher page should render a Streamlit file_uploader"
    assert any(
        any(
            keyword.arg == "max_upload_size"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 25
            for keyword in call.keywords
        )
        for call in file_uploaders
    )
