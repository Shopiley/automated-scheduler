from input_data import input_data
from timetable import Timetable


class Utility:

    @staticmethod
    def print_input_data():
        print(f"Nostgrp={input_data.nostudentgroup} Noteachers={len(input_data.faculties)} "
              f"daysperweek={input_data.days} hoursperday={input_data.hours}")
        
        for i in range(input_data.nostudentgroup):
            student_group = input_data.student_groups[i]
            print(f"{student_group.id} {student_group.name}")
            
            for j in range(student_group.no_courses):
                print(f"{student_group.courseIDs[j]} {student_group.hours_required[j]} hrs {student_group.teacherIDS[j]}")
            print("")

        for i in range(len(input_data.faculties)):
            teacher = input_data.faculties[i]
            print(f"{teacher.faculty_id} {teacher.name} {teacher.courseID}")
    
    @staticmethod
    def print_slots():
        days = input_data.days
        hours = input_data.hours
        nostgrp = input_data.nostudentgroup
        
        # print(Timetable.__dict__)

        print("----Slots----")
        for i in range(days * hours * nostgrp):
            slot = Timetable.slot[i]
            if slot is not None:
                print(f"{i}- {slot.student_group.name} {slot.course_id} {slot.faculty_id}")
            # else:
            #     print("Free Period")
            
            if (i + 1) % (hours * days) == 0:
                print("******************************")
