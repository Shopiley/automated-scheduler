from typing import List
from entitities.course import Course
from entitities.faculty import Faculty


class StudentGroup:
    def __init__(self, id: str, name:str, courseIDs: List[str], teacherIDS: List[str], hours_required:List[int]):
        self.id = id
        self.name = name
        self.no_students= len(id)
        self.courseIDs = courseIDs
        self.no_courses = len(courseIDs)
        self.teacherIDS = teacherIDS    # consider teacherID
        self.hours_required = hours_required
        # self.school = school

    # def __repr__(self):
    #     return f"StudentGroup(id={self.id}, name={self.name}, courseIDs={self.courseIDs}, teacherIDS={self.teacherIDS}, hours_required={self.hours_required})"


