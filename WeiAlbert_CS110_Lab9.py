## Name:Albert Wei

## ID:2607700

"""

This code was my own work, it was written without consulting any sources

outside of those approved by the instructor. 

Initial: AW

"""
#task 1
number_list = list(range(1, 21))

i = 2 
while i < len(number_list):
    del number_list[i]
    i += 2  

print("After removing every 3rd element:", number_list)

#task 2 
grocery = ["oranges", "dragonfruit", "pineapples", "bread", "yogurt"]
electronic = ["headphones", "Speaker", "Scooter", "tv", "keyboard"]


shopping_list = []

shopping_list = [grocery[0], grocery[1], electronic[4], electronic[3]]

print("Shopping list:", shopping_list)

