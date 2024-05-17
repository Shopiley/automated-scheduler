class TimeSlot:
    def __init__(self, id:int, day: str, start_time: str, end_time: str, available: bool):
        self.id = id  
        self.day = day  
        self.start_time = start_time 
        self.end_time = end_time 
        self.available = available  