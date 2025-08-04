import random
from typing import List
import copy
import numpy as np
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

from old_GA.Gene import Gene
import utils
from entitities.Class import Class
from input_data import input_data

# population initialization using input_data
class GeneticAlgorithm:
    def __init__(self, input_data, pop_size: int, mutation_rate: float, crossover_rate: float):
        self.desired_fitness = 0
        self.rooms = input_data.rooms
        self.timeslots = input_data.create_time_slots(no_hours_per_day=input_data.hours, no_days_per_week=input_data.days, day_start_time=9)
        self.student_groups = input_data.student_groups
        self.courses = input_data.courses
        self.events_list, self.events_map = self.create_events()
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()  # List to hold all chromosomes

    def create_events(self):
        events_list = []
        event_map = {}
        idx = 0
        for student_group in self.student_groups:
            for i in range(student_group.no_courses):
                self.hourcount = 1 
                while self.hourcount <= student_group.hours_required[i]:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    event_map[idx] = event
                    idx += 1
                    self.hourcount += 1
        return events_list, event_map
    
    def initialize_population(self):
        population = [] 
        for _ in range(self.pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return np.array(population)

    def create_chromosome(self):
        chromosome = np.empty((len(self.rooms), len(self.timeslots)), dtype=object)
        unassigned_events = []
        for idx, event in enumerate(self.events_list):
            valid_slots = self.get_valid_slots(event)
            if valid_slots:
                row, col = random.choice(valid_slots)
                chromosome[row, col] = idx
            else:
                unassigned_events.append(idx)
        return chromosome

    def get_valid_slots(self, event):
        valid_slots = []
        course = input_data.getCourse(event.course_id)
        for room_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                for i in range(len(self.timeslots)):
                    if self.is_slot_available(room_idx, i):
                        valid_slots.append((room_idx, i))
        return valid_slots

    def is_slot_available(self, room_idx, timeslot_idx):
        return self.population[room_idx][timeslot_idx] is None

    def is_room_suitable(self, room, course):
        return room.room_type == course.required_room_type
    
    def ensure_valid_solution(self, mutant_vector):
        return mutant_vector
    
    def count_non_none(self, arr):
        return np.count_nonzero(arr != None)
    
    def remove_previous_event_assignment(self, chromosome, gene):
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                if chromosome[room_idx][timeslot_idx] == gene:
                    chromosome[room_idx][timeslot_idx] = None
                    return
                
    def hamming_distance(self, chromosome1, chromosome2):
        return np.sum(chromosome1.flatten() != chromosome2.flatten())

    def calculate_population_diversity(self):
        total_distance = 0
        comparisons = 0
        assert len(self.population) == self.pop_size, f"Population size mismatch: expected {self.pop_size}, got {len(self.population)}"
        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):
                total_distance += self.hamming_distance(self.population[i], self.population[j])
                comparisons += 1
        return total_distance / comparisons if comparisons > 0 else 0
    
    def evaluate_fitness(self, chromosome):
        penalty = 0
        cost = 0
        penalty += self.check_room_constraints(chromosome)
        penalty += self.check_student_group_constraints(chromosome)
        penalty += self.check_lecturer_availability(chromosome)
        return penalty + cost   
    
    def check_room_constraints(self, chromosome):
        point = 0
        for room_idx in range(len(self.rooms)):
            room = self.rooms[room_idx]
            for timeslot_idx in range(len(self.timeslots)):
                class_event = self.events_map.get(chromosome[room_idx][timeslot_idx])
                if class_event is not None:
                    course = input_data.getCourse(class_event.course_id)
                    if room.room_type != course.required_room_type or class_event.student_group.no_students > room.capacity:
                        point += 1
        return point
    
    def check_student_group_constraints(self, chromosome):
        penalty = 0
        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            student_group_watch = set()
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    student_group = class_event.student_group
                    if student_group.id in student_group_watch:
                        penalty += 1
                    else:
                        student_group_watch.add(student_group.id)
        return penalty
    
    def check_lecturer_availability(self, chromosome):
        penalty = 0
        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            lecturer_watch = set()
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    faculty_id = class_event.faculty_id
                    if faculty_id in lecturer_watch:
                        penalty += 1
                    else:
                        lecturer_watch.add(faculty_id)
        return penalty
    
    def tournament_selection(self, k: int) -> np.ndarray:
        if self.pop_size < k:
            raise ValueError("Not enough population members to perform selection.")
        tournament_indices = np.random.choice(self.pop_size, k, replace=False)        
        tournament_individuals = [self.population[i] for i in tournament_indices]
        fitness_values = [self.evaluate_fitness(individual) for individual in tournament_individuals]
        winner_index = np.argmin(fitness_values)
        return tournament_individuals[winner_index]
    
    def rank_based_selection(self):
        fitnesses = [self.evaluate_fitness(individual) for individual in self.population]
        sorted_indices = np.argsort(fitnesses)
        sorted_population = np.array(self.population)[sorted_indices]
        ranks = np.arange(1, len(self.population) + 1)
        total_rank = np.sum(ranks)
        probabilities = ranks[::-1] / total_rank
        selected_indices = np.random.choice(len(sorted_population), size=self.pop_size, p=probabilities)
        return sorted_population[selected_indices]

    def roulette_wheel_selection(self):
        fitnesses = np.array([self.evaluate_fitness(individual) for individual in self.population])
        shift_fitness = fitnesses - fitnesses.min() + 1e-6
        inverted_fitnesses = 1 / shift_fitness
        total_fitness = np.sum(inverted_fitnesses)
        probabilities = inverted_fitnesses / total_fitness
        selected_indices = np.random.choice(len(self.population), size=self.pop_size, p=probabilities)
        return np.array(self.population)[selected_indices]

    def multi_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray, points: int = 2) -> np.ndarray:
        num_rows, num_cols = parent1.shape
        child = np.empty_like(parent1)
        crossover_points = sorted(np.random.choice(range(num_cols), points, replace=False))
        toggle = True
        start = 0
        for point in crossover_points:
            if toggle:
                child[:, start:point] = parent1[:, start:point]
            else:
                child[:, start:point] = parent2[:, start:point]
            toggle = not toggle
            start = point
        if toggle:
            child[:, start:] = parent1[:, start:]
        else:
            child[:, start:] = parent2[:, start:]
        return self.fix_event_assignment(child, num_rows, num_cols)
    
    def fix_event_assignment(self, child, num_rows, num_cols):
        assigned_events = set()
        for room_idx in range(num_rows):
            for timeslot_idx in range(num_cols):
                event = child[room_idx, timeslot_idx]
                if event is not None:
                    if event in assigned_events:
                        child[room_idx, timeslot_idx] = None
                    else:
                        assigned_events.add(event)
        empty_slots = list(np.argwhere(child == None))
        all_events = set(self.events_map.keys())
        missing_events = list(all_events - assigned_events)
        for event in missing_events:
            if not empty_slots:
                break
            slot_idx = random.randint(0, len(empty_slots) - 1)
            row, col = empty_slots[slot_idx]
            child[row, col] = event
            empty_slots.pop(slot_idx)
        return self.ensure_valid_solution(child)
        
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        if random.random() < self.mutation_rate:
            room_idx1 = random.randint(0, len(self.rooms) - 1)
            timeslot_idx1 = random.randint(0, len(self.timeslots) - 1)
            room_idx2 = random.randint(0, len(self.rooms) - 1)
            timeslot_idx2 = random.randint(0, len(self.timeslots) - 1)
            temp = chromosome[room_idx1, timeslot_idx1]
            chromosome[room_idx1, timeslot_idx1] = chromosome[room_idx2, timeslot_idx2]
            chromosome[room_idx2, timeslot_idx2] = temp
        return chromosome
    
    def crossover_population(self, parent_population: np.ndarray) -> np.ndarray:
        offspring_population = []
        for _ in range(self.pop_size // 2):
            parent1 = parent_population[random.randint(0, self.pop_size - 1)]
            parent2 = parent_population[random.randint(0, self.pop_size - 1)]
            if random.random() < self.crossover_rate:
                child1 = self.multi_point_crossover(parent1, parent2, points=2)
                child2 = self.multi_point_crossover(parent2, parent1, points=2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            offspring_population.append(child1)
            offspring_population.append(child2)
        if len(offspring_population) < self.pop_size:
            additional_individuals = self.pop_size - len(offspring_population)
            for _ in range(additional_individuals):
                offspring_population.append(self.create_chromosome())
        return np.array(offspring_population)

    def mutate_population(self, population: np.ndarray) -> np.ndarray:
        mutated_population = []
        for individual in population:
            mutated_individual = self.mutate(individual)
            mutated_population.append(mutated_individual)
        return np.array(mutated_population)

    def run(self, max_generations: int):
        self.initialize_population()
        fitness_history = []
        diversity_history = []
        for generation in range(max_generations):
            parent_population = [self.tournament_selection(k=3) for _ in range(self.pop_size)]
            offspring_population = self.crossover_population(np.array(parent_population))
            mutated_population = self.mutate_population(offspring_population)
            self.population = mutated_population
            try:
                best_solution = min(self.population, key=self.evaluate_fitness)
                best_fitness = self.evaluate_fitness(best_solution)
                fitness_history.append(best_fitness)
            except Exception as e:
                print(f"Error during fitness evaluation in generation {generation+1}: {e}")
                return None
            population_diversity = self.calculate_population_diversity()
            diversity_history.append(population_diversity)
            if best_fitness <= self.desired_fitness:
                break
        utils.print_all_timetables(best_solution, self.events_map, input_data.days, input_data.hours)
        return best_solution, fitness_history, generation, diversity_history

    def print_timetable(self, individual, student_group, days, hours_per_day, day_start_time=9):
        timetable = [['Free' for _ in range(days)] for _ in range(hours_per_day)]
        for room_idx, room_slots in enumerate(individual):
            for timeslot_idx, event in enumerate(room_slots):
                class_event = self.events_map.get(event)
                if class_event is not None and class_event.student_group.id == student_group.id:
                    day = timeslot_idx // hours_per_day
                    hour = timeslot_idx % hours_per_day
                    if day < days:
                        timetable[hour][day] = f"Course: {class_event.course_id}, Lecturer: {class_event.faculty_id}, Room: {room_idx}"
        print(f"Timetable for Student Group: {student_group.name}")
        print(" " * 15 + " | ".join([f"Day {d+1}" for d in range(days)]))
        print("-" * (20 + days * 15))
        for hour in range(hours_per_day):
            time_label = f"{day_start_time + hour}:00"
            row = [timetable[hour][day] if timetable[hour][day] else "Free" for day in range(days)]
            print(f"{time_label:<15} | " + " | ".join(row))
        print("\n")
        return timetable

    def print_all_timetables(self, individual, days, hours_per_day, day_start_time=9):
        data = []
        student_groups = input_data.student_groups
        for student_group in student_groups:
            timetable = self.print_timetable(individual, student_group, days, hours_per_day, day_start_time)
            rows = []
            for hour in range(hours_per_day):
                time_label = f"{day_start_time + hour}:00"
                row = [time_label] + [timetable[hour][day] for day in range(days)]
                rows.append(row)
            data.append({"student_group": student_group, "timetable": rows})
        return data

ga = GeneticAlgorithm(input_data, pop_size=50, mutation_rate=0.6, crossover_rate=0.7)
best_solution, fitness_history, generation, diversity_history = ga.run(200)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("GA Timetable Output"),
    html.Div(id='tables-container')
])

@app.callback(
    Output('tables-container', 'children'),
    [Input('tables-container', 'n_clicks')]
)
def render_tables(n_clicks):
    all_timetables = ga.print_all_timetables(best_solution, input_data.days, input_data.hours, 9)
    tables = []
    for timetable_data in all_timetables:
        table = dash_table.DataTable(
            columns=[{"name": "Time", "id": "Time"}] + [{"name": f"Day {d+1}", "id": f"Day {d+1}"} for d in range(input_data.days)],
            data=[dict(zip(["Time"] + [f"Day {d+1}" for d in range(input_data.days)], row)) for row in timetable_data["timetable"]],
            style_cell={
                'textAlign': 'center',
                'height': 'auto',
                'whiteSpace': 'normal',
            },
            style_data_conditional=[
                {'if': {'column_id': 'Day 1'}, 'backgroundColor': 'lightblue', 'color': 'black'},
                {'if': {'column_id': 'Day 2'}, 'backgroundColor': 'lightgreen', 'color': 'black'},
                {'if': {'column_id': 'Day 3'}, 'backgroundColor': 'lavender', 'color': 'black'},
                {'if': {'column_id': 'Day 4'}, 'backgroundColor': 'lightcyan', 'color': 'black'},
                {'if': {'column_id': 'Day 5'}, 'backgroundColor': 'lightyellow', 'color': 'black'},
            ],
            tooltip_data=[
                {f"Day {d+1}": {'value': 'Room info goes here', 'type': 'markdown'} for d in range(input_data.days)} for row in timetable_data["timetable"]
            ],
            tooltip_duration=None
        )
        tables.append(html.Div([html.H3(f"Timetable for {timetable_data['student_group'].name}"), table]))
    return tables

if __name__ == '__main__':
    app.run_server(debug=True)
