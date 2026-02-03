"""
## Name: Albert Wei
## ID:2607700
## Initial below the statement:
AW
This code was my own work, it was written without consulting any sources
outside of those approved by the instructor. 
Initial: 
AW
Homework 4 Instructions
- Your code MUST run successfully without syntax errors to receive full points.
- Submit this .py file to Canvas and save it with the .py extension.
- Make sure to test each condition once to ensure it works. 
- For example, if a condition involves being divisible by 3 and 5, test numbers that satisfy one, both, and neither condition.
- Tip: To rerun a question multiple times, comment out the rest of the code using cmd+/ (Mac) or ctrl+/ (Windows/Linux).
When done, uncomment to test the program again.
"""

"""
[25 pts] Question 1: Grocery Discounts
A grocery store is offering discounts based on the number of items a customer purchases:
15% discount if the number of items is divisible by both 3 and 5.
10% discount if divisible by 3 only.
5% discount if divisible by 5 only.
No discount if not divisible by 3 or 5.
Write a program that asks the user to input the number of items purchased and prints which discount they get.
"""

# Question 1: Grocery Discounts
num_items = int(input("Enter the number of items purchased: "))

if num_items % 3 == 0 and num_items % 5 == 0:
    print("You get a 15% discount.")
elif num_items % 3 == 0:
    print("You get a 10% discount.")
elif num_items % 5 == 0:
    print("You get a 5% discount.")
else:
    print("No discount available.")

# Question 2: Bank Transactions
amount = float(input("Enter the amount to withdraw: "))

if amount > 1000:
    print("Large transaction, approval required.")
elif amount >= 500:
    print("Transaction in review.")
elif amount < 10:
    pass  # No output for small transactions
else:
    print("Transaction approved.")

# Question 3: Grade System
grade = int(input("Enter your grade: "))

if grade >= 90:
    print("A")
elif grade >= 80:
    print("B")
elif grade >= 70:
    print("C")
elif grade >= 60:
    print("D")
else:
    print("F")

