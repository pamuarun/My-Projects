# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:37:17 2024

@author: Arun Teja
"""

import random
#Generate a random number between 1 to 100
number = random.randint(1,100)
print("Welcome To The Number Guessing Game")
print("im done with selecting the number")
print("Try to guess it")
#Intializing a variable
attempts = 0
#Main Game
while True:
    #Taking input from the user
    guess = int(input("Enter Your Number :"))
    attempts +=1#Incrementing
    #Using if elif else for multiple cases
    if guess < number:
        print("Too Low")
    elif guess > number:
        print("Too High")
    else:
        print(f"Hooray! you guessed the number {attempts} attempts")
        break
