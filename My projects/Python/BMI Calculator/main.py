import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import pandas as pd

class HealthAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Health Assistant")
        self.root.configure(bg='#023e8a')

        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TButton", foreground="yellow", background="green", padding=(10, 5), font=("Helvetica", 12, "bold"))
        style.map("TButton", foreground=[('active', 'black'), ('pressed', 'black')])

        style.configure("TLabel", foreground="green", font=("Helvetica", 20, "bold"))
        style.configure("TEntry", foreground="#e63946", bordercolor="green", font=("Helvetica", 20, "bold"))
        style.configure("TText", foreground="#e63946", background="#023e8a", font=("Helvetica", 20, "bold"))

        # Custom heading frame style
        style.configure("Heading.TFrame", background="#023e8a")
        style.configure("Heading.TLabel", foreground="#e63946", font=("Helvetica", 16, "bold"))

        # Medication Reminder Heading
        medication_heading_frame = ttk.Frame(root, style="Heading.TFrame")
        medication_heading_frame.grid(row=0, column=0, pady=10, sticky="ew")
        ttk.Label(medication_heading_frame, text="Medication Reminder", style="Heading.TLabel").pack()

        # Medication Reminder
        ttk.Label(root, text="Patient Name:", style="TLabel").grid(row=1, column=0, sticky="e")
        ttk.Label(root, text="Medication Name:", style="TLabel").grid(row=2, column=0, sticky="e")
        ttk.Label(root, text="Reminder Time:", style="TLabel").grid(row=3, column=0, sticky="e")

        self.patient_name_entry = ttk.Entry(root, width=30, style="TEntry")
        self.medication_name_entry = ttk.Entry(root, width=30, style="TEntry")
        self.reminder_time_entry = ttk.Entry(root, width=30, style="TEntry")

        self.patient_name_entry.grid(row=1, column=1)
        self.medication_name_entry.grid(row=2, column=1)
        self.reminder_time_entry.grid(row=3, column=1)

        ttk.Button(root, text="Set Reminder", command=self.set_medication_reminder, style="TButton").grid(row=4, column=0, columnspan=2, pady=10)

        # Patient Records Heading
        patient_records_heading_frame = ttk.Frame(root, style="Heading.TFrame")
        patient_records_heading_frame.grid(row=5, column=0, pady=10, sticky="ew")
        ttk.Label(patient_records_heading_frame, text="Patient Records", style="Heading.TLabel").pack()

        # Patient Records
        self.patient_records_text = tk.Text(root, height=10, width=50, foreground="#ffb703", background="#023047", font=("Helvetica", 12, "bold"))
        self.patient_records_text.grid(row=6, column=0, columnspan=2)
        ttk.Button(root, text="Save Record", command=self.save_patient_record, style="TButton").grid(row=7, column=0, columnspan=2, pady=10)

        # Appointment Scheduling Heading
        appointment_heading_frame = ttk.Frame(root, style="Heading.TFrame")
        appointment_heading_frame.grid(row=8, column=0, pady=10, )
        ttk.Label(appointment_heading_frame, text="Appointment Scheduling", style="Heading.TLabel").pack()

        # Appointment Scheduling
        ttk.Label(root, text="Appointment Date and Time:", style="TLabel").grid(row=9, column=0, sticky="e")

        self.appointment_datetime_entry = ttk.Entry(root, width=30, style="TEntry")
        self.appointment_datetime_entry.grid(row=9, column=1)

        ttk.Button(root, text="Schedule Appointment", command=self.schedule_appointment, style="TButton").grid(row=10, column=0, columnspan=2, pady=10)

        ttk.Button(root, text="Export to Excel", command=self.export_records_to_excel, style="TButton").grid(row=11, column=0, columnspan=2, pady=10)

    def set_medication_reminder(self):
        patient_name = self.patient_name_entry.get()
        medication_name = self.medication_name_entry.get()
        reminder_time_str = self.reminder_time_entry.get()

        try:
            reminder_time = datetime.strptime(reminder_time_str, "%Y-%m-%d %H:%M:%S")
            current_time = datetime.now()

            if reminder_time > current_time:
                time_difference = reminder_time - current_time
                seconds_difference = time_difference.total_seconds()

                messagebox.showinfo("Reminder Set", f"Medication Reminder set for {patient_name} in {int(seconds_difference)} seconds.")
            else:
                messagebox.showwarning("Invalid Time", "Please choose a future time for the reminder.")
        except ValueError:
            messagebox.showerror("Invalid Time Format", "Please enter the time in the format YYYY-MM-DD HH:MM:SS.")

    def save_patient_record(self):
        patient_record = self.patient_records_text.get("1.0", "end-1c")

        self.patient_records_text.insert("end", patient_record + "\n")

        messagebox.showinfo("Record Saved", "Patient record saved successfully.")

    def schedule_appointment(self):
        appointment_datetime_str = self.appointment_datetime_entry.get()

        try:
            appointment_datetime = datetime.strptime(appointment_datetime_str, "%Y-%m-%d %H:%M:%S")
            current_datetime = datetime.now()

            if appointment_datetime > current_datetime:
                messagebox.showinfo("Appointment Scheduled", f"Appointment scheduled for {appointment_datetime_str}.")
            else:
                messagebox.showwarning("Invalid DateTime", "Please choose a future date and time for the appointment.")
        except ValueError:
            messagebox.showerror("Invalid DateTime Format", "Please enter the date and time in the format YYYY-MM-DD HH:MM:SS.")

    def export_records_to_excel(self):
        records_text = self.patient_records_text.get("1.0", "end-1c").strip().split("\n")

        df = pd.DataFrame({"Patient Record": records_text})

        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file_name = f"health_records_{current_datetime}.xlsx"

        df.to_excel(excel_file_name, index=False)

        messagebox.showinfo("Export Successful", f"Health records exported to {excel_file_name}.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HealthAssistantApp(root)
    root.mainloop()
