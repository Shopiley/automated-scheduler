class Testclass:

    def __init__(self) -> None:
        self.days = 5
        self.hours = 8
        self.hourcount = 1
        self.subject_no = 0
        self.cnt = 0

        self.slot = [(None)]*self.days*self.hours

        for student_group in range(self.days):
            periods = 0
            for i in range(student_group.no_courses):
                self.hourcount = 1
                while self.hourcount <= student_group.hours_required[i]:
                    self.slot[self.cnt] = (student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    self.hourcount += 1
                    self.cnt += 1
                    periods += 1

            while periods < self.days*self.hours:
                periods += 1
                self.cnt += 1
