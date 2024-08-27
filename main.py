import tkinter as tk
from tkinter import messagebox

# Sample Data (dataset)
dataset = [
    [337, 118, 4, 4.5, 4.5, 9.65, 1, 0.92],
    [324, 107, 4, 4, 4.5, 8.87, 1, 0.76],
    [316, 104, 3, 3, 3.5, 8, 1, 0.72],
    [322, 110, 3, 3.5, 2.5, 8.67, 1, 0.8],
    [314, 103, 2, 2, 3, 8.21, 0, 0.65],
    [330, 115, 5, 4.5, 3, 9.34, 1, 0.9],
    [321, 109, 3, 3, 4, 8.2, 1, 0.75],
    [308, 101, 2, 3, 4, 7.9, 0, 0.68],
    [302, 102, 1, 2, 1.5, 8, 0, 0.5],
    [323, 108, 3, 3.5, 3, 8.6, 0, 0.45],
    [325, 106, 3, 3.5, 4, 8.4, 1, 0.52],
    [327, 111, 4, 4, 4.5, 9, 1, 0.84],
    [328, 112, 4, 4, 4.5, 9.1, 1, 0.78],
    [307, 109, 3, 4, 3, 8, 1, 0.62],
    [311, 104, 3, 3.5, 2, 8.2, 1, 0.61]
]


# Function to calculate the chance of admission
def calculate_chance():
    try:
        gre = int(gre_score_entry.get())
        toefl = int(toefl_score_entry.get())
        university_rating = int(university_rating_entry.get())
        sop = float(sop_entry.get())
        lor = float(lor_entry.get())
        cgpa = float(cgpa_entry.get())
        research = int(research_entry.get())

        # Find the closest match in the dataset for simplicity
        closest_match = None
        smallest_diff = float('inf')
        for row in dataset:
            diff = abs(row[0] - gre) + abs(row[1] - toefl) + abs(row[2] - university_rating) + abs(row[3] - sop) + abs(
                row[4] - lor) + abs(row[5] - cgpa) + abs(row[6] - research)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_match = row

        chance_of_admit = closest_match[7]

        # Display the custom message box with a positive or negative message
        if chance_of_admit >= 0.75:
            show_custom_message(f"Congratulations! Your Chance of Admission is {chance_of_admit:.2f}", "Success!",
                                "green", positive=True)
        else:
            show_custom_message(f"Sorry! Your Chance of Admission is {chance_of_admit:.2f}", "Low Chance", "red",
                                positive=False)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")


def show_custom_message(message, title, color, positive):
    # Create a custom top-level window
    custom_msg = tk.Toplevel(root)
    custom_msg.title(title)
    custom_msg.configure(bg=color)

    # Set size and position
    custom_msg.geometry("400x200")

    # Message Label
    msg_label = tk.Label(custom_msg, text=message, font=("Arial", 14, "bold"), bg=color, fg="white")
    msg_label.pack(pady=20)

    # Additional positive quote
    if positive:
        quote = "Keep up the great work and aim high!"
    else:
        quote = "Don’t be discouraged. Keep working hard and you’ll get there!"

    quote_label = tk.Label(custom_msg, text=quote, font=("Arial", 12, "italic"), bg=color, fg="white")
    quote_label.pack(pady=10)

    # Close Button
    close_button = tk.Button(custom_msg, text="OK", command=custom_msg.destroy, font=("Arial", 12, "bold"), bg="white",
                             fg=color)
    close_button.pack(pady=10)


# Tkinter setup
root = tk.Tk()
root.title("Chance of Admission Calculator")

# Set the background color for the main window
root.configure(bg="light blue")

# Font settings
label_font = ("Arial", 12, "bold")
entry_font = ("Arial", 12)
button_font = ("Arial", 14, "bold")

# Labels and Entries with updated size and font
tk.Label(root, text="Graduate Record Examination Score:", font=label_font, bg="light blue").grid(row=0, column=0, padx=10, pady=10)
gre_score_entry = tk.Entry(root, width=30, font=entry_font, bg="white")
gre_score_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Test of English as a Foreign Language Score:", font=label_font, bg="light blue").grid(row=1, column=0, padx=10, pady=10)
toefl_score_entry = tk.Entry(root, width=30, font=entry_font, bg="white")
toefl_score_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="University Rating:", font=label_font, bg="light blue").grid(row=2, column=0, padx=10, pady=10)
university_rating_entry = tk.Entry(root, width=30, font=entry_font, bg="white")
university_rating_entry.grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text=" Statement of Purpose and Letter of Recommendation Strength:", font=label_font, bg="light blue").grid(row=3, column=0, padx=10, pady=10)
sop_entry = tk.Entry(root, width=30, font=entry_font, bg="white")
sop_entry.grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="LOR:", font=label_font, bg="light blue").grid(row=4, column=0, padx=10, pady=10)
lor_entry = tk.Entry(root, width=30, font=entry_font, bg="white")
lor_entry.grid(row=4, column=1, padx=10, pady=10)

tk.Label(root, text="Undergraduate GPA:", font=label_font, bg="light blue").grid(row=5, column=0, padx=10, pady=10)
cgpa_entry = tk.Entry(root, width=30, font=entry_font, bg="white")
cgpa_entry.grid(row=5, column=1, padx=10, pady=10)

tk.Label(root, text="Research Experience:", font=label_font, bg="light blue").grid(row=6, column=0, padx=10, pady=10)
research_entry = tk.Entry(root, width=30, font=entry_font, bg="white")
research_entry.grid(row=6, column=1, padx=10, pady=10)

# Submit Button with increased size and updated font
submit_button = tk.Button(root, text="Submit", command=calculate_chance, font=button_font, bg="dark grey", fg="white",
                          width=15, height=2)
submit_button.grid(row=7, columnspan=2, pady=20)

# Run the Tkinter event loop
root.mainloop()
