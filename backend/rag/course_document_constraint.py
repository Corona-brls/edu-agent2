"""Pure parsing helpers for explicit course-document constraints."""

from __future__ import annotations

import re
import unicodedata


class CourseDocumentConstraintError(ValueError):
    """An explicit course-document constraint could not be resolved safely."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


_EXPLICIT_TITLE_TRIGGER_RE = re.compile(r"只(?:依据|根据)")
_EXPLICIT_TITLE_RE = re.compile(
    r"只(?:依据|根据)\s*(?:当前课程(?:发布的)?\s*)?《(?P<title>[^《》]*)》"
)
_BOOK_TITLE_RE = re.compile(r"《([^《》]*)》")
_EXPLICIT_TITLE_MAX_CHARS = 160


def normalize_course_document_title(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def has_explicit_course_document_constraint_syntax(user_goal: str) -> bool:
    """Recognize valid or malformed strong syntax before generic intent routing."""

    normalized = unicodedata.normalize("NFKC", str(user_goal or ""))
    return bool(_EXPLICIT_TITLE_TRIGGER_RE.search(normalized)) and (
        "《" in normalized or "》" in normalized
    )


def parse_explicit_course_document_title(user_goal: str) -> str | None:
    """Return the single explicit title, or fail closed for malformed strong syntax."""

    normalized = unicodedata.normalize("NFKC", str(user_goal or ""))
    matches = list(_EXPLICIT_TITLE_RE.finditer(normalized))
    if not matches:
        if has_explicit_course_document_constraint_syntax(normalized):
            raise CourseDocumentConstraintError("COURSE_DOCUMENT_CONSTRAINT_INVALID")
        return None
    titles = list(_BOOK_TITLE_RE.finditer(normalized))
    if len(matches) != 1 or len(titles) != 1:
        raise CourseDocumentConstraintError("COURSE_DOCUMENT_CONSTRAINT_INVALID")
    title = normalize_course_document_title(matches[0].group("title"))
    if not title or len(title) > _EXPLICIT_TITLE_MAX_CHARS:
        raise CourseDocumentConstraintError("COURSE_DOCUMENT_CONSTRAINT_INVALID")
    return title
