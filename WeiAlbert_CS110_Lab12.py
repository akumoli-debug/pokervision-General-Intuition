## Name:Albert Wei  

## ID:2607700

"""

This code was my own work, it was written without consulting any sources

outside of those approved by the instructor. 

Initial: AW

"""
from graphics import GraphWin, Circle, Point, Rectangle

class ShapeManager:
    def __init__(self, window_title, window_width, window_height):

        self.win = GraphWin(window_title, window_width, window_height)
        self.shapes = [] 
        
    def add_circle(self, center, radius, color):
        circle = Circle(Point(center[0], center[1]), radius)
        circle.setFill(color)
        circle.draw(self.win)
        self.shapes.append(circle)
        
    def add_rectangle(self, bottom_left, top_right, color):
        rectangle = Rectangle(Point(bottom_left[0], bottom_left[1]), Point(top_right[0], top_right[1]))
        rectangle.setFill(color)
        rectangle.draw(self.win)
        self.shapes.append(rectangle)
        
    def change_color(self, index, color):
        shape = self.shapes[index]
        shape.setFill(color)
        
    def delete_shape(self, index):

        shape = self.shapes.pop(index)
        shape.undraw()
        
    def close_window(self):
    
        self.win.getMouse()  
        self.win.close()  
        
shape_manager = ShapeManager("Shape Manager", 400, 400)


shape_manager.add_circle((120, 300), 50, "red")  
shape_manager.add_circle((120, 300), 30, "green")  
shape_manager.add_rectangle((130, 250), (230, 300), "blue")  

shape_manager.close_window()
