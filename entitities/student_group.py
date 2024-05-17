from typing import List
from entitities.course import Course
from entitities.faculty import Faculty


class StudentGroup:
    def __init__(self, id: str, name:str, no_students:int, courses: List[Course], teachers: List[Faculty], hours_required:List[int]):
        self.id = id
        self.name = name
        self.no_students= no_students
        self.courses = courses
        self.no_courses = len(courses)
        self.teachers = teachers    # consider teacherID
        self.hours_required = hours_required

    
    