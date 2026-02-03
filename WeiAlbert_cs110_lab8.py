## Name:Albert Wei

## ID:2607700

"""

This code was my own work, it was written without consulting any sources

outside of those approved by the instructor. 

Initial: AW

"""
import math 
def circle_info(radius1):
    def valid_radius(radius):
        return isinstance(radius, (int, float)) and radius > 0

    def area(radius):
        return math.pi * radius ** 2

    def circumference(radius):
        return 2 * math.pi * radius
    if not valid_radius(radius1):
        return "Invalid radius. Please provide a positive number."
    
    area_value = round(area(radius1), 2)
    circumference_value = round(circumference(radius1), 2)
    
    print("Area:", area_value, "Circumference:", circumference_value)
    return area_value, circumference_value

# Usage
circle_info(10)