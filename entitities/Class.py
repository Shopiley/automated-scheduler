from entitities.course import Course
from entitities.student_group import StudentGroup

class Class:

    def __init__(self, student_group: StudentGroup, faculty_id: str, course: Course):
        self.student_group = student_group
        self.faculty_id = faculty_id
        self.course = course
