from typing import List
from entitities.faculty import Faculty
class Course:
    def __init__(self, name: str, code: str, faculty: List[Faculty], credits: int, student_groupsID: List[str]):
        self.code = code
        self.name = name
        self.faculty = faculty
        self.credits = credits
        self.no_teachers = len(faculty)
        self.student_groupsID = student_groupsID