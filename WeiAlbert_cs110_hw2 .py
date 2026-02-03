## Name: Nini(Wenting) Ye
## ID: 2644357

## Initial below the statement:
"""
This code was my own work, it was written without consulting any
sources
outside of those approved by the instructor.
Initial: NY
"""
###### [45 pts] Question 1: Variables and Types ######
# [15 pts] Write a program to print 10, 10.0, and print the type of each.
x = 10
y = 10.0
print(x,type(x))
print(y,type(y))
# Then convert the float to an integer, print.
a = int(y)
print(a,type(a))
# Convert the integer to a float, print.
b = float(x)
print(b,type(b))
# Convert the integer to a string, print.
c = str(x)
print(c,type(c))

# [10 pts] Print the expression that adds two integers, 5 and 10, thereby resulting in 15.
print(int(10)+int(5))
# Then try concatenating two strings, '5' and '10', and print the result.
print('10'+'5')

# [10 pts] Assign the value 25 to a variable called 'age'.
# Print the value of age and its type.
age = 25
print(age, type(age))
# Reassign 'age' to a new value of 'twenty-five' (a string).
# Print the new value and check its type.
age = 'twenty-five'
print(age, type(age))

# [10 pts] Verify if Python will accept running:
# '100' + 50 - Make any changes if needed. Then use an f-string to print the result.
##### '100' + 50 will not run because it shown TypeError: can only concatenate str (not "int") to str, integer cant operate with string at same time
##### Changes:
result = int('100')+ 50
print(f"The result is {result}")

###### [15 pts] Question 2: Tuples ######
# [7 pts] Assign values to three variables a, b, and c in a single line using tuple assignment.
# Print them in a single line, separated by commas.
a, b, c = 10, 20, 30
print(f” {a}, {b}, {c}”) # type: ignore

# [8 pts] Print them with custom formatting using the format() method, where a is left-aligned in 10 spaces and b and c are right-aligned in 10 spaces.
print("{0:<10} {1:>10} {2:>10}".format(a, b, c))

###### [15 pts] Question 3: Inputs ######
# Prompt the user to enter their name.
# Use \n to print a greeting on two lines. Additionally, use \t to print their name indented with a tab.
name = input("Enter your name: ")
print(f"Hello!\n\t{name}!")

######  [25 pts] Question 4: Written answers (provide 1-2 sentences each) ###### 
# Written answers should be submitted in a Word document or PDF, not this file. 