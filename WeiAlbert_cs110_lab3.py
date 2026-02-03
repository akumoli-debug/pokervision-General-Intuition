## Name: Albert Wei
## ID:2607700
"""
This code was my own work, it was written without consulting any sources
outside of those approved by the instructor. 
Initial: AW
"""
#right justification
print('{:>3}'.format('x'))
print('{:>4}'.format('xxx'))
print('{:>5}'.format('xxxxxx'))
print('{:>3}'.format('x'))
print('{:>3}'.format('x'))
print('{:>3}'.format('x'))
print('{:>3}'.format('x'))
#task 2
POUNDS_TO_GRAMS = 453.592
GRAMS_TO_KILOGRAMS = 1000
pounds = float(input("Enter weight in pounds: "))
grams = pounds * POUNDS_TO_GRAMS
kilograms = grams / GRAMS_TO_KILOGRAMS
print(f"{pounds:.2f} pounds is {grams:.3f} grams (={kilograms:.3f} kilograms).")

