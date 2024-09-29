# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:06:26 2024

@author: Arun Teja
"""

# Step 1: Initialize variables
balance = 5000  # Initial balance
pin = "1234"    # Default PIN for security
attempts = 3

# Step 2: PIN Verification
while attempts > 0:
    entered_pin = input("Enter your 4-digit PIN: ")
    if entered_pin == pin:
        break
    else:
        attempts -= 1
        print(f"Incorrect PIN. You have {attempts} attempt(s) left.")
        
if attempts == 0:
    print("Too many incorrect attempts. Access denied.")
else:
    # Step 3: ATM Menu Loop
    while True:
        print("\nWelcome to the ATM!")
        print("1. Check Balance")
        print("2. Deposit")
        print("3. Withdraw")
        print("4. Exit")
        
        choice = input("Please select an option (1-4): ")

        # Step 4: Handling user choices
        if choice == "1":
            print(f"Your current balance is: ₹{balance}")
        elif choice == "2":
            deposit_amount = float(input("Enter the amount to deposit: ₹"))
            if deposit_amount > 0:
                balance += deposit_amount
                print(f"₹{deposit_amount} has been deposited. New balance: ₹{balance}")
            else:
                print("Invalid amount. Please try again.")
        elif choice == "3":
            withdraw_amount = float(input("Enter the amount to withdraw: ₹"))
            if 0 < withdraw_amount <= balance:
                balance -= withdraw_amount
                print(f"₹{withdraw_amount} has been withdrawn. New balance: ₹{balance}")
            elif withdraw_amount > balance:
                print("Insufficient balance. Please try again.")
            else:
                print("Invalid amount. Please try again.")
        elif choice == "4":
            print("Thank you for using the ATM. Have a great day!")
            break
        else:
            print("Invalid choice. Please select a valid option.")