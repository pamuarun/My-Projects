# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:34:44 2024

@author: Arun Teja
"""

# Function to perform basic arithmetic operations
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Error! Division by zero."
    else:
        return x / y

# Main Function To Perform the opration
def calculator():
    print("............Mini Calculator............")

    # Taking inputs from the user
    num1 = float(input("Enter 1st number: "))
    num2 = float(input("Enter 2nd number: "))

    # Operations to be performed
    print("Press 1 for addition")
    print("Press 2 for subtraction")
    print("Press 3 for multiplication")
    print("Press 4 for division")

    # The choice is taking from user input
    choice = input("Enter a number between 1-4: ")

    # Using if-elif-else to perform the  operation
    if choice == '1':
        print(f"The addition of {num1} and {num2} is: {add(num1, num2)}")
    elif choice == '2':
        print(f"The subtraction of {num1} and {num2} is: {subtract(num1, num2)}")
    elif choice == '3':
        print(f"The multiplication of {num1} and {num2} is: {multiply(num1, num2)}")
    elif choice == '4':
        print(f"The division of {num1} and {num2} is: {divide(num1, num2)}")
    else:
        print("Invalid input! Please enter a valid number between 1-4.")

# Calling the main function
calculator()
