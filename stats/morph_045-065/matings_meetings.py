# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:57:28 2025

@author: rensk
"""

import matplotlib.pyplot as plt

# Initialize lists
generations = []
matings = []
meetings = []

# Read the file
with open("C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/run 2/extra.txt", "r") as f:
    lines = f.readlines()

gen, mat, meet, count = None, None, None, None
for line in lines:
    line = line.strip()
    if line.startswith("Generation:"):
        gen = int(line.split(":")[1])
    elif line.startswith("Mating:"):
        mat = int(line.split(":")[1])
    elif line.startswith("Meeting:"):
        meet = int(line.split(":")[1])
    # When we have all three, save them
    if gen is not None and mat is not None and meet is not None:
        generations.append(gen)
        matings.append(mat)
        meetings.append(meet)
        gen, mat, meet, count = None, None, None, None  # reset for next block

# Compute percentage of meetings that end in mating
percentage = [m / mt * 100 if mt != 0 else 0 for m, mt in zip(matings, meetings)]

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot percentage on left y-axis
ax1.plot(generations, percentage, color="green", marker="o", label="% Matings")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Percentage (%)", color="green")
ax1.tick_params(axis="y", labelcolor="green")

# Plot number of meetings on right y-axis
ax2 = ax1.twinx()
ax2.plot(generations, meetings, color="red", marker="x", label="Meetings")
ax2.plot(generations, matings, color="blue", marker="s", label= "Matings")
ax2.set_ylabel("Number", color="black")  # general label
ax2.tick_params(axis="y", labelcolor="black")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("Percentage of Matings and Number of Matings and Meetings per Generation")
plt.show()