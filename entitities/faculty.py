from typing import List
from entitities.course import Course


class Faculty:
    def __init__(self, id:str, name:str, department:str, courses: List[Course]):
        self.faculty_id = id
        self.name = name
        self.department = department
        self.courses = courses