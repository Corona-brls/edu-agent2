"""Activity statistics and timeline queries."""

from __future__ import annotations

from typing import Any

from backend.storage.logger import LearningLogger


class ActivityService:
    def __init__(self, storage: LearningLogger) -> None:
        self.storage = storage

    def teacher_activity(
        self,
        *,
        teacher: dict[str, Any],
        course_id: int | None = None,
        class_id: int | None = None,
    ) -> dict[str, Any]:
        role = str(teacher["role"]).lower()
        if role == "teacher":
            classes = self.storage.classes_for_teacher(int(teacher["id"]), course_id=course_id)
            if class_id is not None:
                classes = [item for item in classes if int(item["id"]) == int(class_id)]
            if not classes:
                raise PermissionError("教师只能查看自己任课班级的活动统计。")
            class_ids = [int(item["id"]) for item in classes]
        else:
            class_ids = [class_id] if class_id is not None else None
        return self.storage.learning_activity_summary(course_id=course_id, class_ids=class_ids)

    def student_timeline(
        self,
        *,
        requester: dict[str, Any],
        student_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        role = str(requester["role"]).lower()
        target = int(student_id or requester["id"])
        if role == "student" and target != int(requester["id"]):
            raise PermissionError("学生不能查看其他学生的学习轨迹。")
        if role == "teacher" and not self.storage.teacher_can_access_user(int(requester["id"]), target):
            raise PermissionError("教师只能查看所属班级学生的学习轨迹。")
        return self.storage.learning_timeline(user_id=target, limit=limit)
