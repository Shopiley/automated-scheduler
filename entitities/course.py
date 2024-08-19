from typing import List
# import PhysicalSpaceTypes

class Course:
    def __init__(self, name: str, code: str, credits: int, student_groupsID: List[str], facultyId: str):
        self.code = code
        self.name = name
        self.credits = credits
        self.student_groupsID = student_groupsID
        self.facultyId = facultyId
        # self.physical_space_type_required = physical_space_type_required

    def __repr__(self):
        return f"Course(name={self.name}, code={self.code}, credits={self.credits}, student_groupsID={self.student_groupsID})"
