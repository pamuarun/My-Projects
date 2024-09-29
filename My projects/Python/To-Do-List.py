# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:38:58 2024

@author: Arun Teja
"""

# Step 1: Define an empty list to hold tasks
to_do_list = []
 
# Step 2: Function to display the menu
def display_menu():
    print("\nTo-Do List Application")
    print("1. View Tasks")
    print("2. Add a Task")
    print("3. Mark Task as Complete")
    print("4. Delete a Task")
    print("5. Exit")

# Step 3: Function to view tasks
def view_tasks():
    if not to_do_list:
        print("Your to-do list is empty.")
    else:
        print("\nYour Tasks:")
        for idx, task in enumerate(to_do_list, 1):
            status = "✔" if task['completed'] else "✘"
            print(f"{idx}. {task['name']} [{status}]")

# Step 4: Function to add a task
def add_task():
    task_name = input("Enter the task: ")
    task = {'name': task_name, 'completed': False}
    to_do_list.append(task)
    print(f"Task '{task_name}' added!")

# Step 5: Function to mark a task as complete
def mark_task_complete():
    view_tasks()
    task_num = int(input("Enter the number of the task to mark as complete: ")) - 1
    if 0 <= task_num < len(to_do_list):
        to_do_list[task_num]['completed'] = True
        print(f"Task '{to_do_list[task_num]['name']}' marked as complete!")
    else:
        print("Invalid task number.")

# Step 6: Function to delete a task
def delete_task():
    view_tasks()
    task_num = int(input("Enter the number of the task to delete: ")) - 1
    if 0 <= task_num < len(to_do_list):
        removed_task = to_do_list.pop(task_num)
        print(f"Task '{removed_task['name']}' deleted!")
    else:
        print("Invalid task number.")

# Step 7: Main loop to interact with the user
while True:
    display_menu()
    choice = input("Choose an option (1-5): ")

    if choice == '1':
        view_tasks()
    elif choice == '2':
        add_task()
    elif choice == '3':
        mark_task_complete()
    elif choice == '4':
        delete_task()
    elif choice == '5':
        print("Exiting the To-Do List. Goodbye!")
        break
    else:
        print("Invalid choice, please try again.")