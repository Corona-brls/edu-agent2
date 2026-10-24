import pandas as pd

from frontend.pages.teacher import _safe_csv_bytes


def test_safe_csv_export_escapes_formula_injection():
    df = pd.DataFrame(
        [
            {"代表问题": "=cmd|'/C calc'!A0", "次数": 1},
            {"代表问题": "+SUM(1,2)", "次数": 1},
            {"代表问题": "普通问题", "次数": 1},
        ]
    )

    csv_text = _safe_csv_bytes(df).decode("utf-8-sig")

    assert "'=cmd" in csv_text
    assert "'+SUM" in csv_text
    assert "普通问题" in csv_text
