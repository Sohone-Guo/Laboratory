
class Vector:
    
    def __init__(self, x=0, y=0):
        self.x = x 
        self.y = y 
        
    def __repr__(self):
        return "Vectors({}, {})".format(self.x, self.y)
    
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    
    # def __str__(self):
    #     return "vector({}, {})".format(self.x, self.y)

if __name__ == "__main__":
    a = Vector(3,4)
    b = Vector(1,2)
    print(a+b) # Vectors(4, 6)
        