import tkinter as tk
from tkinter import ttk, messagebox
import random
import string
import pyperclip

class PasswordGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Password Generator")

        self.length_label = ttk.Label(root, text="Password Length:")
        self.length_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.length_var = tk.IntVar(value=12)
        self.length_entry = ttk.Entry(root, textvariable=self.length_var)
        self.length_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.lower_var = tk.IntVar(value=1)
        self.lower_check = ttk.Checkbutton(root, text="Include Lowercase", variable=self.lower_var)
        self.lower_check.grid(row=1, column=0, columnspan=2, pady=5, sticky="w")

        self.upper_var = tk.IntVar(value=1)
        self.upper_check = ttk.Checkbutton(root, text="Include Uppercase", variable=self.upper_var)
        self.upper_check.grid(row=2, column=0, columnspan=2, pady=5, sticky="w")

        self.digit_var = tk.IntVar(value=1)
        self.digit_check = ttk.Checkbutton(root, text="Include Digits", variable=self.digit_var)
        self.digit_check.grid(row=3, column=0, columnspan=2, pady=5, sticky="w")

        self.special_var = tk.IntVar(value=1)
        self.special_check = ttk.Checkbutton(root, text="Include Special Characters", variable=self.special_var)
        self.special_check.grid(row=4, column=0, columnspan=2, pady=5, sticky="w")

        self.generate_button = ttk.Button(root, text="Generate Password", command=self.generate_password)
        self.generate_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(root, textvariable=self.password_var, state="readonly")
        self.password_entry.grid(row=6, column=0, columnspan=2, pady=10)

        self.copy_button = ttk.Button(root, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_button.grid(row=7, column=0, columnspan=2, pady=10)

    def generate_password(self):
        length = self.length_var.get()
        include_lower = bool(self.lower_var.get())
        include_upper = bool(self.upper_var.get())
        include_digits = bool(self.digit_var.get())
        include_special = bool(self.special_var.get())

        if length <= 0:
            messagebox.showwarning("Warning", "Password length must be greater than 0.")
            return

        characters = ""
        if include_lower:
            characters += string.ascii_lowercase
        if include_upper:
            characters += string.ascii_uppercase
        if include_digits:
            characters += string.digits
        if include_special:
            characters += string.punctuation

        if not any((include_lower, include_upper, include_digits, include_special)):
            messagebox.showwarning("Warning", "Please select at least one character type.")
            return

        password = ''.join(random.choice(characters) for _ in range(length))
        self.password_var.set(password)

    def copy_to_clipboard(self):
        password = self.password_var.get()
        pyperclip.copy(password)
        messagebox.showinfo("Success", "Password copied to clipboard.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PasswordGenerator(root)
    root.mainloop()
