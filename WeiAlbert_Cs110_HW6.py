"""
## Name: Albert Wei
## ID:2607700
## Initial below the statement:
This code was my own work, it was written without consulting any sources
outside of those approved by the instructor.
Initial:AW
"""
"""
Homework 6 Instructions
Your code MUST run successfully without syntax errors to receive full points.
Submit this .py file to Canvas and save it with the .py extension.
"""

#####[30 pts] Question 1: Road Trip Tracker ###### - LISTS

# Step 1
def road_trip_tracker():
    stops = []
    while True:
        city = input("Enter a city: ")
        stops.append(city)
        cont = input("Do you want to add another city? (yes/no): ")
        if cont.lower() != 'yes':
            break
    
    if "Atlanta" in stops:
        print("Atlanta is in your road trip stops!")
    else:
        print("Atlanta is not in your road trip stops!")

# Step 2
def packing_suitcase():
    suitcase = ["T-shirt", "Jeans", "Socks", "T-shirt", "Sneakers", "Toothbrush", "Socks", "Hat", "Charger", "Towel", "T-shirt", "Shampoo"]
    print(f"Original suitcase items: {suitcase}")
    
    while True:
        action = input("Do you want to add or remove items from your suitcase? (add/remove/none): ").lower()
        if action == "add":
            item = input("Enter the item to add: ")
            suitcase.append(item)
        elif action == "remove":
            item = input("Enter the item to remove: ")
            if item in suitcase:
                suitcase.remove(item)
            else:
                print(f"{item} is not in your suitcase.")
        elif action == "none":
            break
        else:
            print("Invalid input. Please enter 'add', 'remove', or 'none'.")

    unique_suitcase = []
    for item in suitcase:
        if item not in unique_suitcase:
            unique_suitcase.append(item)
    
    print(f"Updated suitcase without duplicates: {unique_suitcase}")

# Step 3
def frequent_city():
    visited_cities = []
    while True:
        city = input("Enter a city you've visited: ")
        visited_cities.append(city)
        cont = input("Do you want to add another city? (yes/no): ")
        if cont.lower() != 'yes':
            break
    
    city_count = {}
    for city in visited_cities:
        if city in city_count:
            city_count[city] += 1
        else:
            city_count[city] = 1

    most_visited_city = None
    max_visits = 0
    for city, count in city_count.items():
        if count > max_visits:
            most_visited_city = city
            max_visits = count

    if list(city_count.values()).count(max_visits) > 1:
        print("All cities were visited the same number of times.")
    else:
        print(f"The most frequently visited city is {most_visited_city} with {max_visits} visits.")

road_trip_tracker()
packing_suitcase()
frequent_city()


#####
# [10 pts] Step 3: Find the most frequently visited city from a list of travel stops.


# Ask the user to input cities visited multiple times. Prompt the user to type 'yes' when done. Store the cities in a list. 
# Find the most frequently visited city from the list of stops.
# Hint: There is an inbuilt function that can be used to count the number of times an item appears in a list.
stops = []


# Using an f-string, print the most frequently visited city and the number of times it was visited.
# If the frequency of all the cities is the same, print "All cities were visited the same number of times."




"""
Don't remove this comment or the print statement after it.
"""

print("--------- End of Question 1 ---------")  # Do not remove this line


##### [30 pts] Question 2: Tic-Tac-Toe Winner Checker ##### - 2D Lists and Functions

# [15 pts] Step 1: Create a function that checks the winner of a Tic-Tac-Toe game given a 3x3 matrix.
# The matrix will contain 'X', 'O', or an empty string '' representing unmarked spaces.
# You can remane the functions of you wish.

#Hint: Thinking about the problem in terms of rows, columns, and diagonals can help you solve it.

def check_winner(board):

    return None  # Replace with your implementation

#Input board for testing
Input_board = [
    ["X", "O", "X"],
    ["O", "X", "O"],
    ["X", "", "O"]
]

# Display Input board to the user. The board should look like a tic-tac-toe board.
# Hint: There is a function that can be used to concatenate elements of a list into a single string.



# Call the function created earlier and with the input board,  print the result. You can do this in your function or outside of it.


## Step 1: Tic-Tac-Toe Winner Checker
def check_winner(board):
    # Check rows, columns, and diagonals
    for row in board:
        if row[0] == row[1] == row[2] != '':
            return f"Winner: {row[0]}"
    
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != '':
            return f"Winner: {board[0][col]}"
    
    if board[0][0] == board[1][1] == board[2][2] != '':
        return f"Winner: {board[0][0]}"
    
    if board[0][2] == board[1][1] == board[2][0] != '':
        return f"Winner: {board[0][2]}"
    
    return "No winner yet!"

# Step 2: User input for Tic-Tac-Toe
def user_board_input():
    print("Enter your Tic-Tac-Toe board (3x3):")
    board = []
    for i in range(3):
        row = input(f"Enter row {i + 1} (separate with spaces): ").split()
        board.append(row)
    
    print("Your Tic-Tac-Toe board:")
    for row in board:
        print(" ".join(row))
    
    # Check for winner
    print(check_winner(board))

# Test board input
Input_board = [["X", "O", "X"], ["O", "X", "O"], ["X", "", "O"]]
print("Input board:")
for row in Input_board:
    print(" ".join(row))
print(check_winner(Input_board))

user_board_input()


"""
Don't remove this comment or the print statement after it.
"""

print("--------- End of Question 2 ---------")  # Do not remove this line


##### [25 pts] Question 3: Rock, Paper, Scissors Game ##### - Functions

#[5 pts] Step 1: Create a function that randomly selects one of the three choices: 'rock', 'paper', or 'scissors'. For convenience, you can use (r, p, s) as valid inputs.
# Hint: You can use random.choice() to select a random element from a list.

def computer_choice():

    return None  # Replace with your implementation


# [10 pts] Step 2: Create a function that compares the player's choice and the computer's choice.
# The function should determine the winner or if it's a tie.
# Hint: Remember to use if-elif-else statements to cover all possible outcomes.

def determine_winner():

    return None  # Replace with your implementation


# [10 pts] Step 3: Create the main function that allows the player to enter their choice.
# Ensure the player's input is one of the valid choices ('rock', 'paper', or 'scissors'). For convenience, you can use (r, p, s) as valid inputs.
# The function should display both the player's and computer's choices along with the winner.
# Hint: Use the functions you created in steps 1 and 2.

def play_game():

    return None  # Replace with your implementation

# Call the main function to start the game.
import random

# Step 1: Randomly select choice
def computer_choice():
    return random.choice(['rock', 'paper', 'scissors'])

# Step 2: Compare choices and determine winner
def compare_choices(player, computer):
    if player == computer:
        return "It's a tie!"
    elif (player == 'rock' and computer == 'scissors') or \
         (player == 'scissors' and computer == 'paper') or \
         (player == 'paper' and computer == 'rock'):
        return "You win!"
    else:
        return "Computer wins!"

# Step 3: Main game function
def play_rps():
    player_choice = input("Enter your choice (rock, paper, or scissors): ").lower()
    while player_choice not in ['rock', 'paper', 'scissors']:
        player_choice = input("Invalid input. Please enter rock, paper, or scissors: ").lower()
    
    comp_choice = computer_choice()
    print(f"Computer chose: {comp_choice}")
    
    result = compare_choices(player_choice, comp_choice)
    print(result)

# Start the game
play_rps()



"""
Don't remove this comment or the print statement after it.
"""
print("--------- End of Question 3  ---------")  # Do not remove this line




###### [15 pts] Question 4: Written answers (provide 1-2 sentences each) ######
# Written answers should be submitted in a Word document or PDF, not this file.


