"""Course visibility helpers."""

from __future__ import annotations

from backend.storage.logger import LearningLogger


def visible_courses(storage: LearningLogger, user: dict) -> list[dict]:
    role = str(user["role"]).lower()
    if role == "admin":
        return storage.courses()
    if role == "teacher":
        return storage.courses_for_teacher(int(user["id"]))
    return storage.courses_for_user(int(user["id"]))


def visible_classes(storage: LearningLogger, user: dict, course_id: int | None = None) -> list[dict]:
    role = str(user["role"]).lower()
    if role == "admin":
        return storage.classes(course_id=course_id)
    if role == "teacher":
        return storage.classes_for_teacher(int(user["id"]), course_id=course_id)
    return storage.classes_for_user(int(user["id"]), course_id=course_id)

