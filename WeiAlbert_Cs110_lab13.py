from graphics import *

class ShapeInteractionManager:
    def __init__(self):
        self.shapes = []
        self.win = GraphWin("Shape Interaction Manager", 500, 500)
        self.message = Text(Point(250, 20), "Click on two points to draw a line.")
        self.message.draw(self.win)

    def add_colorline(self, point1, point2, color):
        line = Line(point1, point2)
        line.setOutline(color)
        line.draw(self.win)
        self.shapes.append(line)

    def add_triangle(self, points, color):
        triangle = Polygon(points)
        triangle.setFill(color)
        triangle.draw(self.win)
        self.shapes.append(triangle)

    def add_text_input(self, location, prompt):
        prompt_text = Text(Point(location.getX(), location.getY() - 20), prompt)
        prompt_text.draw(self.win)
        text_input = Entry(location, 10)
        text_input.draw(self.win)
        return text_input

    def close(self):
        self.win.close()

def main():
    manager = ShapeInteractionManager()

    
    point1 = manager.win.getMouse()
    point2 = manager.win.getMouse()
    manager.add_colorline(point1, point2, "blue")

    
    manager.message.setText("Click on three points to draw a triangle.")
    points = []
    for _ in range(3):
        points.append(manager.win.getMouse())
    manager.add_triangle(points, "green")

   
    manager.message.setText("Enter a color to change the triangle.")
    color_input = manager.add_text_input(Point(250, 400), "Enter color:")
    color = color_input.getText()
    if color:
        for shape in manager.shapes:
            if isinstance(shape, Polygon):
                shape.setFill(color)

    
    manager.message.setText("Click anywhere to quit.")
    manager.win.getMouse()


    manager.close()

if __name__ == "__main__":
    main()
