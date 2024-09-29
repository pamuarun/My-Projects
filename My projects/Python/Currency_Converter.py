# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:19:15 2024

@author: Arun Teja
"""

# Step 1: Define exchange rates
exchange_rates = {
    'USD': 1.0,  # Base currency
    'EUR': 0.85,
    'GBP': 0.75,
    'INR': 74.0,
    'JPY': 110.0
}

# Step 2: Function to display available currencies
def display_currencies():
    print("Available currencies:")
    for currency in exchange_rates.keys():
        print(currency)

# Step 3: Function to convert currency
def convert_currency(amount, from_currency, to_currency):
    if from_currency in exchange_rates and to_currency in exchange_rates:
        # Convert amount to USD first, then to the target currency
        amount_in_usd = amount / exchange_rates[from_currency]
        converted_amount = amount_in_usd * exchange_rates[to_currency]
        return converted_amount
    else:
        return None

# Step 4: Main program loop
while True:
    print("\nSimple Currency Converter")
    display_currencies()
    
    from_currency = input("Enter the currency you want to convert from (or type 'exit' to quit): ").upper()
    if from_currency == 'EXIT':
        print("Exiting the currency converter. Goodbye!")
        break

    to_currency = input("Enter the currency you want to convert to: ").upper()
    amount = float(input("Enter the amount you want to convert: "))
    
    # Step 5: Perform conversion
    converted_amount = convert_currency(amount, from_currency, to_currency)
    if converted_amount is not None:
        print(f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}.")
    else:
        print("Invalid currency input. Please try again.")