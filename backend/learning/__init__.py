"""Learning content helpers shared by later course-context features."""

from backend.learning.context import (
    CourseReference,
    CourseReferenceError,
    load_course_references,
)
from backend.learning.source_catalog import (
    CatalogValidationError,
    KnowledgeSourceCatalog,
    KnowledgeSourceEntry,
    load_source_catalog,
)

__all__ = [
    "CatalogValidationError",
    "CourseReference",
    "CourseReferenceError",
    "KnowledgeSourceCatalog",
    "KnowledgeSourceEntry",
    "load_course_references",
    "load_source_catalog",
]
"""Shared learning services and bounded model runtime."""
