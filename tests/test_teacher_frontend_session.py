from pathlib import Path

from streamlit.testing.v1 import AppTest


TEACHER_PAGE = Path(__file__).resolve().parents[1] / "frontend" / "pages" / "teacher.py"


def test_anonymous_direct_teacher_page_renders_login_gate_without_business_data():
    page = AppTest.from_file(str(TEACHER_PAGE), default_timeout=10).run()

    assert not page.exception
    assert page.title[0].value == "Teacher Console"
    assert any("log in" in item.value.lower() for item in page.info)
    assert not page.tabs
