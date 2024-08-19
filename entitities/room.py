rooms ={
    "small_classroom": 13,
    "medium_classroom": 20,
    "large_classroom": 8,
}
class Room:
    def __init__(self, Id: str, name:str, capacity:int, room_type:str):
        self.Id = Id
        self.name = name
        self.capacity = capacity
        self.room_type = room_type

    def __repr__(self):
        return f"Room(Id={self.Id}, name={self.name}, capacity={self.capacity}, room_type={self.room_type})"

# an inventory class, with list 
# test = Room("1", "room1", 20, "small_classroom")      