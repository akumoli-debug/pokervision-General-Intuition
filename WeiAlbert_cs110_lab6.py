## Name:Albert Wei

## ID: 2607700

"""

This code was my own work, it was written without consulting any sources

outside of those approved by the instructor. 

Initial: AW

"""
#Task 1 Part 1
count = 1
while count <= 10:
    print(count)
    count += 1  
#Task 1 Part 2
count = 1
while count <= 50:
    if count % 5 == 0:  
        print(count)
    count += 1  
#Task 2 
total_sum = 0

while True:
    x = int(input("Enter an integer (negative number to stop): "))
    if x < 0:
        break 
    total_sum += x  

print("Sum of positive numbers:", total_sum)

