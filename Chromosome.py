from Gene import Gene
from entitities.Class import Class
from input_data import input_data
import copy
import random

from timetable import Timetable


"""
A chromosome represents a complete timetable for all student groups.
It's a collection of genes, one for each (timeslot-student-room) combination.
The size of the chromosome will be dependent on the number of (student groups * time slots) and rooms.
"""

class Chromosome:
    # Static variables equivalent to static fields in Java
    # crossoverrate = None
    # mutationrate = None
    # hours = None
    # days = None
    # nostgrp = None

    def __init__(self):
        # Initialize the static variables from input_data
        self.crossoverrate = input_data.cross_over_rate
        self.mutationrate = input_data.mutation_rate
        self.hours = input_data.hours
        self.days = input_data.days
        self.nostgrp = len(input_data.student_groups)

        self.fitness = 0
        self.point = 0
        self.gene = [None] * self.nostgrp

        for i in range(self.nostgrp):
            self.gene[i] = Gene(i)

        # Calculate fitness
        self.fitness = self.get_fitness()

    def deep_clone(self):
        return copy.deepcopy(self)

    def get_fitness(self):  # the smaller the fitness, the better the solution
        self.point = 0

        for i in range(self.hours * self.days):
            teacherlist = []

            for j in range(self.nostgrp):
                slot = None

                if Timetable.slot[self.gene[j].slotno[i]] is not None:
                    slot = Timetable.slot[self.gene[j].slotno[i]]

                if slot is not None:
                    if slot.faculty_id in teacherlist:
                        self.point += 1
                    else:
                        teacherlist.append(slot.faculty_id)

        self.fitness = 1 - (self.point / ((self.nostgrp - 1.0) * self.hours * self.days))
        self.point = 0
        return self.fitness

    def print_time_table(self):
        for i in range(self.nostgrp):
            status = False
            l = 0

            while not status:
                if Timetable.slot[self.gene[i].slotno[l]] is not None:
                    print(f"Batch {Timetable.slot[self.gene[i].slotno[l]].student_group.name} Timetable-")
                    status = True
                l += 1

            for j in range(self.days):
                for k in range(self.hours):
                    if Timetable.slot[self.gene[i].slotno[k + j * self.hours]] is not None:
                        print(f"{Timetable.slot[self.gene[i].slotno[k + j * self.hours]].course_id} ", end="")
                    else:
                        print("*FREE* ", end="")
                print("")
            print("\n\n\n end")

    def print_chromosome(self):
        for i in range(self.nostgrp):
            for j in range(self.hours * self.days):
                print(f"{self.gene[i].slotno[j]} ", end="")
            print("")

    def __lt__(self, other):
        return self.fitness > other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

# test_chromosome = Chromosome()
# print(test_chromosome.print_time_table())
# print(test_chromosome.print_chromosome())