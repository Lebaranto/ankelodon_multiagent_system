import math

def calculate_area(radius):
    """Calculate the area of a circle given its radius."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * (radius ** 2)

radius = 5
area = calculate_area(radius)
print(area)