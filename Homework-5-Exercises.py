"""
## Name: Albert Wei
## ID:2607700
## Initial below the statement:
AW
This code was my own work, it was written without consulting any sources outside of those approved by the instructor. 
Initial: AW

Homework 5 Instructions
- Your code MUST run successfully without syntax errors to receive full points.
- Submit this .py file to Canvas and save it with the .py extension. 
- Tip: To rerun a question multiple times, comment out the rest of the code using cmd+/ (Mac) or ctrl+/ (Windows/Linux).
When done, uncomment to test the program again.
"""

"""
[25 pts] Question 1: Playlist Maker
- Write a Python program that asks the user how many songs they want to add to their playlist.
- For each song, ask the user to enter the song title
- Also, ask the user to enter the length in minutes (as a float, e.g., 3.5 for 3 minutes and 30 seconds).
- Print each song as it's added, showing the song number, title, and length.
After all songs are entered, print a final message showing how many songs were added and the total playlist length (rounded to two decimal places).

"""
# Playlist Maker using functions

def get_song_details():
    song_title = input("Enter a song title: ")
    song_length = float(input("How long is the song in minutes (float)? "))
    return song_title, song_length

def display_song(count, song_title, song_length):
    print(str(count) + " " + song_title + " " + str(song_length) + " minutes")

# Main program
def create_playlist():
    num_songs = int(input("How many songs do you want to add? "))
    total_time = 0
    count = 1

    while count <= num_songs:
        song_title, song_length = get_song_details()
        display_song(count, song_title, song_length)
        total_time += song_length
        count += 1
    
    print("Your playlist is " + str(num_songs) + " songs long and " + str(round(total_time, 2)) + " minutes long!")

create_playlist()


"""
[25 pts] Question 2: Fibonacci Number Finder
- Write a Python program that asks the user for how many Fibonacci numbers they want to see.
- Use a while loop to generate and print the Fibonacci sequence.
- The Fibonacci sequence starts with 0 and 1, and each number after is the sum of the two previous numbers.

"""
# Fibonacci Number Finder using functions

def fibonacci_sequence(n):
    first = 0
    second = 1
    count = 0
    while count < n:
        print(first)
        next_fib = first + second
        first = second
        second = next_fib
        count += 1

# Main program
def get_fibonacci_numbers():
    number = int(input("How many Fibonacci numbers? "))
    fibonacci_sequence(number)

get_fibonacci_numbers()

"""
[25 pts] Question 3: Savings Goal Tracker
- Write a Python program that asks the user for their savings goal.
- Then, in a loop, ask how much they saved each week until they reach or exceed the goal.
- After each entry, print the total saved so far and how much more is needed to reach the goal.
- When the goal is reached, print a congratulatory message.

"""
# Savings Goal Tracker using functions

def track_savings(goal):
    total = 0
    week = 1
    while total < goal:
        saved_this_week = float(input(f"How much did you save in week {week}? "))
        total += saved_this_week
        
        if total < goal:
            print(f"Total saved so far: {total}")
            print(f"You're {goal - total} away from your goal! Keep going!")
        
        week += 1

    print(f"Congratulations! You reached your savings goal of {goal} dollars!")

# Main program
def get_savings_goal():
    goal = float(input("What is your savings goal? "))
    track_savings(goal)

get_savings_goal()

