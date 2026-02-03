"""
## Name:Albert Wei
## ID:2607700
## Initial below the statement:
This code was my own work, it was written without consulting any sources
outside of those approved by the instructor.
Initial:AW
"""
"""
Homework 8 Instructions
Your code MUST run successfully without syntax errors to receive full points.
Submit this .py file to Canvas and save it with the .py extension.
"""
#####[30 pts] Question 1:
'''
You're designing a robot face that blinks its eyes using the graphics.py library. Follow the steps to complete your animated robot.
'''

from graphics import *
import time

# [6 pts] Step 1: Set up the window
# - Create a window titled "Blinking Robot Face"
# - Set size to 400x400 pixels
# - Set coordinate system from (0,0) to (10,10)

win = GraphWin("Blinking Robot Face", 400, 400)
win.setCoords(0.0, 0.0, 10.0, 10.0)

# [6 pts] Step 2: Draw the robot's head
# - Use a gray rectangle from (2,2) to (8,8)

head = Rectangle(Point(2, 2), Point(8, 8))
head.setFill("gray")
head.draw(win)

# [6 pts] Step 3: Add two square eyes
# - Left eye from (3,6.5) to (4,7.5)
# - Right eye from (6,6.5) to (7,7.5)
# - Start with white fill color

left_eye = Rectangle(Point(3, 6.5), Point(4, 7.5))
right_eye = Rectangle(Point(6, 6.5), Point(7, 7.5))
left_eye.setFill("white")
right_eye.setFill("white")
left_eye.draw(win)
right_eye.draw(win)

# [5 pts] Step 4: Add a mouth
# - Use a black rectangle from (4,3) to (6,3.5)

mouth = Rectangle(Point(4, 3), Point(6, 3.5))
mouth.setFill("black")
mouth.draw(win)

# [4 pts] Step 5: Add a red antenna
# - Draw a red circle at (5,8.5) with radius 0.3
# - Connect it to the head with a black line from (5,8.0) to (5,8.5)

antenna_tip = Circle(Point(5, 8.5), 0.3)
antenna_tip.setFill("red")
antenna_tip.draw(win)

antenna_line = Line(Point(5, 8.0), Point(5, 8.5))
antenna_line.setFill("black")
antenna_line.draw(win)

# [3 pts] Step 6: Animate blinking eyes
# - Make both eyes blink (change to black, then white)
# - Repeat this 3 times with a short pause between each (use time.sleep)

for _ in range(3):
    left_eye.setFill("black")
    right_eye.setFill("black")
    time.sleep(0.3)
    left_eye.setFill("white")
    right_eye.setFill("white")
    time.sleep(0.3)

# Wait for user click before closing the window
win.getMouse()
win.close()

print("--------- End of Question1 ---------")  # Do not remove this line


import random

# [11 pts] Step 1: Define the ElectricCar class
class ElectricCar:
    def __init__(self, model):
        self.model = model
        self.battery = 100
        self.distance_traveled = 0

    def race(self):
        if self.battery >= 10:
            distance = random.randint(10, 30)
            self.battery -= 10
            self.distance_traveled += distance
            print(f"{self.model} raced {distance} miles. Battery: {self.battery}%. Total distance: {self.distance_traveled} miles")
        else:
            print(f"{self.model} can't race — battery is too low.")

    def charge(self, amount):
        self.battery = min(self.battery + amount, 100)
        print(f"{self.model} charged up. Battery now {self.battery}%")

    def __str__(self):
        return f"{self.model}: Battery at {self.battery}%, Distance: {self.distance_traveled} miles"

# [8 pts] Step 2: Define the RaceManager class
class RaceManager:
    def __init__(self, race_name):
        self.race_name = race_name

    def start_race(self, car1, car2):
        print(f"Starting race: {self.race_name}")
        while car1.battery >= 10 and car2.battery >= 10:
            car1.race()
            car2.race()
        if car1.distance_traveled > car2.distance_traveled:
            print(f"{car1.model} wins with {car1.distance_traveled} miles!")
        elif car2.distance_traveled > car1.distance_traveled:
            print(f"{car2.model} wins with {car2.distance_traveled} miles!")
        else:
            print("It's a tie!")

# [6 pts] Step 3: Try it out!
car1 = ElectricCar("Zoomer")
car2 = ElectricCar("Bolt")
manager = RaceManager("Sky Track 3000")

manager.start_race(car1, car2)

car1.charge(50)
car2.charge(50)

print(car1)
print(car2)

print("--------- End of Question 2---------")  # Do not remove this line


"""
Don't remove this comment or the print statement after it.
"""

print("--------- End of Question 2---------")  # Do not remove this line


###### [25 pts] Question 3 and 4: Written answers (provide 1-2 sentences each) ######
# Written answers should be submitted in a Word document or PDF, not this file.