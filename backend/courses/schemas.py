from pydantic import BaseModel


class CoursePublic(BaseModel):
    id: int
    name: str
    description: str = ""
    assessment_method: str = ""


class ClassPublic(BaseModel):
    id: int
    course_id: int
    name: str
    term: str = ""
    teacher_id: int | None = None

