import random
from typing import List
import copy
from Gene import Gene
from utils import Utility
from entitities.Class import Class
from input_data import input_data
import numpy as np

# population initialization using input_data
class GeneticAlgorithm:
    def __init__(self, input_data, pop_size: int, mutation_rate: float):
        self.desired_fitness = 0
        self.rooms = input_data.rooms
        self.timeslots = input_data.create_time_slots(no_hours_per_day=input_data.hours, no_days_per_week=input_data.days, day_start_time=9)
        self.student_groups = input_data.student_groups
        self.courses = input_data.courses
        self.events_list, self.events_map = self.create_events()
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()  # List to hold all chromosomes

    def create_events(self):
        events_list = []
        event_map = {}

        idx = 0
        for student_group in self.student_groups:
            # periods = 0
            for i in range(student_group.no_courses):
                self.hourcount = 1 
                while self.hourcount <= student_group.hours_required[i]:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    
                    # Add the event to the index map with the current index
                    event_map[idx] = event
                    idx += 1

                    self.hourcount += 1
                    # self.cnt += 1
                    # periods += 1
        # print(events_list, len(events_list))
        return events_list, event_map
    
    def initialize_population(self):
        population = [] 
        for _ in range(self.pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return np.array(population)

    def create_chromosome(self):
        # 2D NumPy array for chromosome where each row is a room and each column is a time slot
        chromosome = np.empty((len(self.rooms), len(self.timeslots)), dtype=object)

        # return chromosome
        unassigned_events = []
        for idx, event in enumerate(self.events_list):
            valid_slots = []
            course = input_data.getCourse(event.course_id)
            for room_idx, room in enumerate(self.rooms):
                if self.is_room_suitable(room, course):
                    for i in range(len(self.timeslots)):
                        if self.is_slot_available(chromosome, room_idx, i):
                            valid_slots.append((room_idx, i))

            if len(valid_slots) > 0:
                row, col = random.choice(valid_slots)
                chromosome[row, col] = idx
            else:
                unassigned_events.append(idx) #TODO
        # print("unassigned events", unassigned_events)
        return chromosome
    
    def is_slot_available(self, chromosome, room_idx, timeslot_idx):
        # Check if the slot is available (i.e., not already assigned)
        return chromosome[room_idx][timeslot_idx] is None

    def is_room_suitable(self, room, course):
        return room.room_type == course.required_room_type
    
    def ensure_valid_solution(self, mutant_vector):
        # for now
        return mutant_vector
    
    def count_non_none(self, arr):
        # Flatten the 2D array and count elements that are not None
        return np.count_nonzero(arr != None)
    
    def remove_previous_event_assignment(self, chromosome, gene):
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                if chromosome[room_idx][timeslot_idx] is not None and chromosome[room_idx][timeslot_idx] == gene:
                    chromosome[room_idx][timeslot_idx] = None
                    return
                
    def hamming_distance(self, chromosome1, chromosome2):
        return np.sum(chromosome1.flatten() != chromosome2.flatten())

    def calculate_population_diversity(self):
        total_distance = 0
        comparisons = 0

        # Debug: Check the size of population vs. pop_size
        assert len(self.population) == self.pop_size, f"Population size mismatch: expected {self.pop_size}, got {len(self.population)}"
        
        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):
                total_distance += self.hamming_distance(self.population[i], self.population[j])
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0
    
    def evaluate_fitness(self, chromosome):
        penalty = 0
        cost = 0
        
        # Check for hard constraint violations (H1-H5)
        penalty += self.check_room_constraints(chromosome)  # H1
        penalty += self.check_student_group_constraints(chromosome)  # H2
        # penalty += self.check_room_time_conflict(chromosome)  # H3
        penalty += self.check_lecturer_availability(chromosome)  # H4
        # penalty += self.check_valid_timeslot(chromosome)  # H5
        
        # Check for soft constraint violations (S1-S3)
        # cost += self.check_single_event_per_day(chromosome)  # S1
        # cost += self.check_consecutive_timeslots(chromosome)  # S2

        # Fitness is a combination of penalties and costs
        return penalty + cost   
    
    def check_room_constraints(self, chromosome):
        """
        rooms must meet the capacity and type of the scheduled event
        """
        point = 0
        for room_idx in range(len(self.rooms)):
            room = self.rooms[room_idx]
            for timeslot_idx in range(len(self.timeslots)):
                class_event = self.events_map.get(chromosome[room_idx][timeslot_idx])
                if class_event is not None:
                    course = input_data.getCourse(class_event.course_id)
                    # std_grp = input_data.getStudentGroup(class_event.student_grp)
                    # H1: Room capacity and type constraints
                    if room.room_type != course.required_room_type or class_event.student_group.no_students > room.capacity:
                        # print("failed - rooms must meet the capacity and type of the scheduled event")
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
                        # print(f"failed - {student_group.id} has more than one event in a timeslot")
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
                        # print(f"failed - {faculty_id} has more than one event in a timeslot")
                    else:
                        lecturer_watch.add(faculty_id)

        return penalty
    
    def tournament_selection(self, k: int) -> np.ndarray:
        """
        Perform tournament selection.
        Args:
            k: Tournament size (number of individuals in the tournament)
        Returns:
            The selected individual (chromosome) from the population.
        """
        # Step 1: Randomly select k individuals from the population
        if self.pop_size < k:
            raise ValueError("Not enough population members to perform selection.")
        
        tournament_indices = np.random.choice(self.pop_size, k, replace=False)        
        tournament_individuals = [self.population[i] for i in tournament_indices]
        
        # Step 2: Evaluate fitness of each individual in the tournament
        fitness_values = [self.evaluate_fitness(individual) for individual in tournament_individuals]
        
        # Step 3: Select the individual with the best (lowest) fitness
        winner_index = np.argmin(fitness_values)
        winner = tournament_individuals[winner_index]
        
        return winner
    
    
    # def rank_based_selection(self):
    #     # Step 1: Calculate fitness for each individual
    #     fitnesses = [self.evaluate_fitness(individual) for individual in self.population]

    #     # Step 2: Rank individuals based on their fitness (lower fitness is better)
    #     # sorted_population = [x for _, x in sorted(zip(fitnesses, self.population))]
    #     sorted_indices = np.argsort(fitnesses)  # Get sorted indices based on fitness
    #     sorted_population = np.array(self.population)[sorted_indices] 
        
    #     # Step 3: Assign ranks and selection probabilities
    #     ranks = list(range(1, len(self.population) + 1))  # Rank from 1 to population size
    #     total_rank = sum(ranks)  # Sum of ranks
    #     probabilities = [rank / total_rank for rank in ranks]  # Selection probability based on rank
        
    #     # Step 4: Perform selection based on rank probabilities
    #     # selected_individuals = np.random.choice(sorted_population, size=self.pop_size, p=probabilities)
    #     selected_indices = np.random.choice(len(sorted_population), size=self.pop_size, p=probabilities)
    #     selected_individuals = sorted_population[selected_indices]
        
    #     return selected_individuals

    def rank_based_selection(self):
        # Step 1: Calculate fitness for each individual
        fitnesses = [self.evaluate_fitness(individual) for individual in self.population]

        # Step 2: Rank individuals based on their fitness (lower fitness is better)
        sorted_indices = np.argsort(fitnesses)  # Get sorted indices based on fitness (ascending order)
        sorted_population = np.array(self.population)[sorted_indices]

        # Step 3: Assign ranks and selection probabilities (better individuals should have lower rank)
        ranks = np.arange(1, len(self.population) + 1)  # Rank from 1 (best) to population size (worst)
        total_rank = np.sum(ranks)
        
        # Reverse the ranks so the best individuals (lower fitness) have the highest probability
        probabilities = ranks[::-1] / total_rank

        # Step 4: Perform selection based on rank probabilities
        selected_indices = np.random.choice(len(sorted_population), size=self.pop_size, p=probabilities)
        selected_individuals = sorted_population[selected_indices]

        return selected_individuals


    # def roulette_wheel_selection(self):
    #     # Step 1: Calculate fitness for each individual in the population
    #     fitnesses = np.array([self.evaluate_fitness(individual) for individual in self.population])
        
    #     # Step 2: Invert fitness values for minimization problem (lower fitness = better)
    #     # Add a small constant to avoid division by zero for individuals with fitness of 0.
    #     inverted_fitnesses = 1 / (fitnesses + 1e-6)
        
    #     # Step 3: Calculate total inverted fitness and probabilities for each individual
    #     total_fitness = np.sum(inverted_fitnesses)
    #     probabilities = inverted_fitnesses / total_fitness
        
    #     # Step 4: Select individuals based on roulette wheel probabilities
    #     selected_indices = np.random.choice(len(self.population), size=self.pop_size, p=probabilities)
    #     selected_individuals = np.array(self.population)[selected_indices]
        
    #     return selected_individuals


    def roulette_wheel_selection(self):
        # Step 1: Calculate fitness for each individual in the population
        fitnesses = np.array([self.evaluate_fitness(individual) for individual in self.population])
        
        # Step 2: Shift fitness values so they are all positive
        # Since smaller fitness values are better, shift them by subtracting the min value
        shift_fitness = fitnesses - fitnesses.min() + 1e-6  # Add small constant to avoid division by zero
        
        # Step 3: Invert fitness values for selection (lower fitness should have higher probability)
        inverted_fitnesses = 1 / shift_fitness
        
        # Step 4: Calculate total inverted fitness and probabilities
        total_fitness = np.sum(inverted_fitnesses)
        probabilities = inverted_fitnesses / total_fitness
        
        # Step 5: Select individuals based on the roulette wheel probabilities
        selected_indices = np.random.choice(len(self.population), size=self.pop_size, p=probabilities)
        selected_individuals = np.array(self.population)[selected_indices]
        
        return selected_individuals

        
    def multi_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray, points: int = 2) -> np.ndarray:
        """
        Perform a multi-point crossover between two parent chromosomes.
        Args:
            parent1: The first parent chromosome (2D array).
            parent2: The second parent chromosome (2D array).
            points: Number of crossover points (default is 2 for two-point crossover).
        Returns:
            child: The resulting child chromosome (2D array).
        """
        num_rows, num_cols = parent1.shape
        child = np.empty_like(parent1)

        # Choose random crossover points along the rows
        crossover_points = sorted(np.random.choice(range(num_cols), points, replace=False))

        # Use the crossover points to alternate between parent1 and parent2
        toggle = True
        start = 0
        for point in crossover_points:
            if toggle:
                child[:, start:point] = parent1[:, start:point]
            else:
                child[:, start:point] = parent2[:, start:point]
            toggle = not toggle
            start = point

        # Fill in the remaining columns
        if toggle:
            child[:, start:] = parent1[:, start:]
        else:
            child[:, start:] = parent2[:, start:]

        return self.fix_event_assignment(child, num_rows, num_cols)
        
        # return child
    
    def fix_event_assignment(self, child, num_rows, num_cols):
        
        # Step 1: Track which events have been assigned already
        assigned_events = set()  # To track all assigned events in the child

    # Step 3: Track assigned events and check for duplicates or missing events
        for room_idx in range(num_rows):
            for timeslot_idx in range(num_cols):
                event = child[room_idx, timeslot_idx]
                if event is not None:
                    if event in assigned_events:
                        # Step 4: Handle event duplication - remove duplicated events
                        child[room_idx, timeslot_idx] = None
                    else:
                        assigned_events.add(event)

        # Step 5: Fill unassigned slots with missing events
        empty_slots = list(np.argwhere(child == None))
        all_events = set(self.events_map.keys())  # All events
        missing_events = list(all_events - assigned_events)  # Find missing events

        # Fill the empty slots with missing events
        for event in missing_events:
            if not empty_slots:
                break  # If no empty slots left, stop
            # Choose a random empty slot
            slot_idx = random.randint(0, len(empty_slots) - 1)
            row, col = empty_slots[slot_idx]
            
            # Assign event to the chosen empty slot
            child[row, col] = event
            
            # Remove the slot from empty_slots by its index
            empty_slots.pop(slot_idx)

        # Ensure valid child
        child = self.ensure_valid_solution(child)

        return child
        
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform mutation on a given chromosome by swapping two random events.
        Args:
            chromosome: The chromosome to mutate (2D array).
            mutation_rate: Probability of mutation for each gene.
        Returns:
            Mutated chromosome.
        """
        if random.random() < self.mutation_rate:
            room_idx1 = random.randint(0, len(self.rooms) - 1)
            timeslot_idx1 = random.randint(0, len(self.timeslots) - 1)
            room_idx2 = random.randint(0, len(self.rooms) - 1)
            timeslot_idx2 = random.randint(0, len(self.timeslots) - 1)

            # Swap the events between two random positions
            temp = chromosome[room_idx1, timeslot_idx1]
            chromosome[room_idx1, timeslot_idx1] = chromosome[room_idx2, timeslot_idx2]
            chromosome[room_idx2, timeslot_idx2] = temp

        return chromosome
    
    def crossover_population(self, parent_population: np.ndarray) -> np.ndarray:
        offspring_population = []
        
        # Generate new population through crossover
        for _ in range(self.pop_size // 2):  # Assuming crossover generates 2 offspring per pair
            parent1 = parent_population[random.randint(0, self.pop_size - 1)]
            parent2 = parent_population[random.randint(0, self.pop_size - 1)]
            # print(f"parent1: {parent1}, parent2:{parent2}")

            # Perform crossover
            child1 = self.multi_point_crossover(parent1, parent2, points=2)
            child2 = self.multi_point_crossover(parent2, parent1, points=2)

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
        # np.random.seed(seed)
        self.initialize_population()
        fitness_history = []
        diversity_history = []

        for generation in range(max_generations):
            parent_population = [self.tournament_selection(k=3) for _ in range(self.pop_size)]
            # parent_population = self.rank_based_selection()
            # parent_population = self.roulette_wheel_selection()

            # Crossover: Generate offspring via crossover
            offspring_population = self.crossover_population(np.array(parent_population))

            # Mutation: Mutate offspring population
            mutated_population = self.mutate_population(offspring_population)

            # Evaluate fitness and replace the old population with new population
            self.population = mutated_population
            
            try:
                # Evaluate the best solution in the population
                best_solution = min(self.population, key=self.evaluate_fitness)
                best_fitness = self.evaluate_fitness(best_solution)
                
                fitness_history.append(best_fitness)

            except Exception as e:
                print(f"Error during fitness evaluation in generation {generation+1}: {e}")
                return None  # Return early in case of error

            # Measure and track diversity
            population_diversity = self.calculate_population_diversity()
            diversity_history.append(population_diversity)
            # print(f"Best solution for generation {generation+1} has a fitness of: {best_fitness}, Diversity: {population_diversity}")

            if best_fitness <= self.desired_fitness:
                # print("Desired fitness achieved!")
                # print(best_solution, self.count_non_none(best_solution))
                break

        return best_solution, fitness_history, generation, diversity_history

        # self.print_all_timetables(best_solution, input_data.days, input_data.hours)

# import time

pop_size = 53
# F = 0.8
max_generations = 50
# CR = 0.9

ga = GeneticAlgorithm(input_data, pop_size=53, mutation_rate=0.5)


# start_time = time.time()
ga.run(max_generations)
# ga_time = time.time() - start_time

# print(f'Time: {ga_time:.2f}s')