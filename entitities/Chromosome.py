from entitities.Gene import Gene


class Chromosome:
    def __init__(self, num_slots, num_days):
        self.num_slots = num_slots
        self.num_days = num_days
        self.genes = [[Gene() for _ in range(num_days)] for _ in range(num_slots)]

