"""
## Name: Albert Wei
## ID: 2607700

## Initial below the statement:
AW
This code was my own work, it was written without consulting any sources
outside of those approved by the instructor.
Initial:AW
"""
"""
Homework 3 Instructions
Your code MUST run successfully without syntax errors to receive full points.
Submit this .py file to Canvas and save it with the .py extension.
"""
###### [30 pts] Question 1: Travel Budgeting ######
# Write code below. Make sure it runs without error.
# Prompt the user to enter the:
# total cost of flight tickets for the group (integer).
# total cost of the hotel booking for the group (integer).
# the number of travelers.
# the destination for the trip.
flight_cost = int(input("Enter cost of flight tickets for the group."))
hotel_cost = int(input("Enter cost of the hotel booking for the group."))
num_travelers = int(input("Enter number of travelers."))
destination = str(input("Enter destination for the trip."))

#####
# Calculate the total travel cost.
total_cost = flight_cost + hotel_cost

# Calculate cost per traveler using different operations:
# Use integer division to find the whole dollar amount each traveler should pay.
cost_per_traveler_int = total_cost // num_travelers 

# Use floating point division to find the precise amount (including cents) each traveler should pay.
cost_per_traveler_float = total_cost / num_travelers

# Use the remainder to find any leftover dollars that couldn’t be evenly divided.
remaining_amount = total_cost % num_travelers

# Use f-strings to print:
# The total travel cost for going to <name of the destination>.
print(f"The total travel cost for going to {destination} is ${total_cost}.")

# Cost each traveler should pay (integer division).
print(f"Each traveler should pay ${cost_per_traveler_int} in integer.")

# Cost each traveler should pay (floating-point division).
print(f'Each traveler should pay ${cost_per_traveler_float} in float')

# Any remaining dollar amount using the % operator.
print(f"Remaining dollar amount: ${remaining_amount}.")


###### [[30 pts] Question 2: Splitting the Bill ######
# Write code below. Make sure it runs without error. #
# Prompt the user to enter the restaurant name.
# Prompt the user to enter the number of people splitting the bill.
# Prompt the user to enter the total bill amount (integer).
# Prompt the user to enter the tip to be added (e.g., 15 for 15%)
restaurant_name = str(input("Enter the restaurant name: "))
num_people = int(input("Enter the number of people splitting the bill: "))
bill_amount = int(input("Enter the total bill amount: "))
tip_percentage = int(input("Enter the tip percentage: "))

# Convert tip percentage to a decimal
tip_decimal = float(tip_percentage/100)

# Calculate the tip amount
tip_amount = float(bill_amount * tip_decimal)

# Calculate the final total amount
final_total = bill_amount + tip_amount

# Calculate the amount each person owes
cost_per_person = final_total/num_people

# Print the final message using one f-string with the specified format
print(f"Thank you for chooing{restaurant_name}! Your total amount for today is ${final_total}, you selected {tip_percentage}% tip. Each person pays ${cost_per_person} for today's meal.")

###### [20 pts] Question 3: Professional Profile Summary ######
# Write code below. Make sure it runs without error. #
# Prompt the user to enter:
# Your name
# Your age
# Your major
# Your experience with Python or another programming language.
# Your career goals or why you're taking this class.
name = str(input("Enter your name: "))
age =  int(input("Enter your age: "))
major = str(input("Enter your major: "))
experience = str(input("Enter your experience with python or another programming language: "))
goal = str(input("Enter your career goal or why you're taking this class: "))

# Use a multi-line string with triple quotes (can be ''' or """). Use .format() to insert the user inputs into the string. Print the message.
message = '''
Name: {}
Age: {}
Major: {}
Experience: {}
Goal: {}
'''.format(name, age, major, experience, goal)
print(f'{message}')

###### [20 pts] Question 4: Written answers (provide 1-2 sentences each) ######
# Written answers should be submitted in a Word document or PDF, not this file.