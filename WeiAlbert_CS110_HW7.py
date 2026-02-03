"""
## Name: Albert Wei
## ID:2607700
## Initial below the statement:
This code was my own work, it was written without consulting any sources
outside of those approved by the instructor.
Initial:AW
"""
"""
Homework 7 Instructions
Your code MUST run successfully without syntax errors to receive full points.
Submit this .py file to Canvas and save it with the .py extension.
"""

#####[25 pts] Question 1:

playlist = ['Not Like Us', 'Oceans', 'Die With A Smile', 'Taste', 
            'Oscar Winning Tears,', 'Elastic Heart', 'As It Was', 'Bad Habits']

# [4 pts] Step 1: Slice and print the middle 4 songs.
# Hint: Think of where the middle starts and ends.

class Playlist:
    def __init__(self, songs):
        self.songs = songs
    
    def print_middle_4(self):
        middle_4 = self.songs[2:6]
        print("Middle 4 songs:", middle_4)

    def print_even_songs(self):
        even_songs = self.songs[::2]
        print("Songs at even indices:", even_songs)

    def print_reverse_playlist(self):
        reverse_playlist = self.songs[::-1]
        print("Playlist in reverse order:", reverse_playlist)

    def print_last_3_songs(self):
        last_3_songs = self.songs[-3:]
        print("Last 3 songs:", last_3_songs)

    def remove_first_and_last(self):
        middle_playlist = self.songs[1:-1]
        print("Playlist without first and last songs:", middle_playlist)

    def check_bad_habits_in_second_half(self):
        second_half = self.songs[len(self.songs)//2:]
        if 'Bad Habits' in second_half:
            print("Bad Habits is in the second half.")
        else:
            print("Bad Habits is not in the second half.")


# Radar class to handle radar grid operations
class Radar:
    def __init__(self, grid):
        self.grid = grid

    def scan_for_aliens(self):
        detected_aliens = []
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] == 'A':
                    detected_aliens.append((r, c))
                    print(f"Alien detected at ({r}, {c})")
        return detected_aliens

    def scan_row_or_column(self):
        choice = input("Would you like to scan a row or column? ").strip().lower()
        if choice == "row":
            row = int(input("Enter the row number (0-3): "))
            aliens_count = self.grid[row].count('A')
            print(f"{aliens_count} aliens found in row {row}")
        elif choice == "column":
            col = int(input("Enter the column number (0-3): "))
            aliens_count = sum(1 for r in range(len(self.grid)) if self.grid[r][col] == 'A')
            print(f"{aliens_count} aliens found in column {col}")



# [4 pts] Step 2: Print every song at an even index.
# Hint: Even indices are 0, 2, 4, ...

# Your code here:


# [4 pts] Step 3: Print the playlist in reverse order using slicing.

# Your code here:



# [4 pts] Step 4: Print the last 3 songs on the playlist using negative indexing.

# Your code here:



# [4 pts] Step 5: Remove the first and last songs using slicing.
# Print the new playlist without modifying the original one.

# Your code here:



# [5 pts] Step 6: Check if 'Bad Habits' is in the second half of the playlist.
# If yes, print "'Bad Habits' is in the second half!"
# Otherwise, print "Nope, it's in the first half."

# Your code here:




"""
Don't remove this comment or the print statement after it.
"""

print("--------- End of Question 1 ---------")  # Do not remove this line


##### [30 pts] Question 2: 

# You’re simulating a survival grid where each cell may contain a gladiator (G), trap (X), or empty space (-).
# Complete each step to simulate the environment and assess the armor's position.

import random

# [10 pts] Step 1: Create the battlefield.
# - Make a 5x5 2D list.
# - Randomly fill each cell with "G" (gladiator), "X" (trap), or "-" (empty space).
# - Print the grid.

# Your code here:



# [6 pts] Step 2: Ask the user where to place armor.
# - Prompt the user to enter a row (0–4) and a column (0–4).
# - Place an "A" at that location in the grid.
# - Replace whatever was there before.
# - Print the updated grid.

# Your code here:



# [5 pts] Step 3: Count nearby gladiators.
# - Count how many "G" are in the same row as the armor.
# - Print the number in this format: "There are 2 gladiators near the armor!"

# Your code here:



# [5 pts] Step 4: Check for danger!
# - Count how many traps ("X") are in the same column as the armor.
# - If there are 2 or more traps, print "This spot is too dangerous!"
# - Otherwise, print "Armor placed safely."

# Your code here:



# [4 pts] Step 5: Give a report.
# - Count the total number of gladiators ("G") on the grid.
# - Print: "Total gladiators in the arena: X"

# Your code here:


"""
Don't remove this comment or the print statement after it.
"""

print("--------- End of Question 2 ---------")  # Do not remove this line


##### [30 pts] Question 3: 

# You’re scanning a radar screen for aliens in a 4x4 grid.
# Some cells have an alien marked as "A", others are just empty marked with "-".
# Follow the steps to build and operate the radar.

import random

# [10 pts] Step 1: Create the radar grid.
# - Make a 4x4 2D list filled with a mix of "A" (alien) and "-"
# - You can hard-code the list manually or use random placement
# - Print the radar grid to show its current state

# Your code here:
playlist = Playlist(['Not Like Us', 'Oceans', 'Die With A Smile', 'Taste', 'Oscar Winning Tears', 'Elastic Heart', 'As It Was', 'Bad Habits'])
playlist.print_middle_4()
playlist.print_even_songs()
playlist.print_reverse_playlist()
playlist.print_last_3_songs()
playlist.remove_first_and_last()
playlist.check_bad_habits_in_second_half()

# Radar object and operations
radar_grid = [
    ['A', '-', 'A', '-'],
    ['A', '-', '-', 'A'],
    ['A', '-', '-', '-'],
    ['A', 'A', 'A', '-']
]
radar = Radar(radar_grid)
radar.scan_for_aliens()
radar.scan_row_or_column()

# -------------- Simple Graphics Example -------------- #
def display_graphics():
    win = GraphWin("Simple Graphics", 400, 400)
    win.setCoords(0.0, 0.0, 10.0, 10.0)

    # Draw some shapes
    circle = Circle(Point(5, 5), 2)
    circle.setFill("blue")
    circle.draw(win)

    rectangle = Rectangle(Point(1, 1), Point(4, 4))
    rectangle.setFill("green")
    rectangle.draw(win)

    text = Text(Point(5, 8), "Click to close the window")
    text.draw(win)

    # Wait for a mouse click to close the window
    win.getMouse()
    win.close()

# Uncomment to see the graphics window
# display_graphics()


# [10 pts] Step 2: Scan the radar!
# - Write a function that goes through the grid
# - Return a list of all coordinates (row, col) where "A" is found
# - Print each one using this format: "Alien detected at (row, col)"

# Your code here:


# [10 pts] Step 3: Strategic analysis
# - Ask the user if they want to scan a row or a column
# - Ask them for the row number or column number (0 to 3)
# - Count how many aliens are in that row or column
# - Print the result in the format: "3 aliens found in row 2" or "1 alien found in column 1"

# Your code here:



"""
Don't remove this comment or the print statement after it.
"""
print("--------- End of Question 3  ---------")  # Do not remove this line




###### [15 pts] Question 4: Written answers (provide 1-2 sentences each) ######
# Written answers should be submitted in a Word document or PDF, not this file.


