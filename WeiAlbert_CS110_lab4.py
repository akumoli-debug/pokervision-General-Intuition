## Name:Albert Wei
## ID:2607700
"""
This code was my own work, it was written without consulting any sources
outside of those approved by the instructor. 
Initial: AW
"""
# Task 1
x = 5
y = 10
z = 0

# Printing the Boolean results of the given conditions
print(x < y or x >= z)    # a. True
print(x != y and x == z)  # b. False
print(z < y and not (z > x))  # c. True
print(y > x and y > z)    # d. True

# Task 2
# Prompt the user to enter an integer
num = int(input("enter an integer: "))
# Check if the number is zero or non-zero and determine its sign
if num == 0:
    print(f"{num} is zero.")
else:
    if num > 0:
        print(f"{num} is not zero and it is a positive integer.")
    else:
        print(f"{num} is not zero and it is a negative integer.")

