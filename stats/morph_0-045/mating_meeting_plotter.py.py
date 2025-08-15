# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:38:42 2025

@author: rensk
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/mating_meeting_dict.json."
with open(filename, "r") as f:
    all_experiments = json.load(f)


generations = list(range(100))

# Prepare arrays for averaging
all_percentages = []
all_matings = []
all_meetings = []

for exp_data in all_experiments.values():
    matings = np.array(exp_data["matings"])
    meetings = np.array(exp_data["meetings"])
    percentage = np.where(meetings != 0, matings / (meetings + 1e-10)* 100, 0)
    
    all_percentages.append(percentage)
    all_matings.append(matings)
    all_meetings.append(meetings)

# Convert to numpy arrays for easy averaging
all_percentages = np.array(all_percentages)
all_matings = np.array(all_matings)
all_meetings = np.array(all_meetings)

# Compute mean and std
mean_percentage = all_percentages.mean(axis=0)
std_percentage = all_percentages.std(axis=0)

mean_matings = all_matings.mean(axis=0)
std_matings = all_matings.std(axis=0)

mean_meetings = all_meetings.mean(axis=0)
std_meetings = all_meetings.std(axis=0)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Percentage line with shaded variation
ax1.plot(generations, mean_percentage, color="green", label="% Matings/Meetings")
ax1.fill_between(generations, mean_percentage - std_percentage, mean_percentage + std_percentage,
                 color="green", alpha=0.2)

# Matings and meetings lines with shaded variation
ax2.plot(generations, mean_meetings, color="blue", label="Meetings")
ax2.fill_between(generations, mean_meetings - std_meetings, mean_meetings + std_meetings,
                 color="blue", alpha=0.2)

ax2.plot(generations, mean_matings, color="orange", label="Matings")
ax2.fill_between(generations, mean_matings - std_matings, mean_matings + std_matings,
                 color="orange", alpha=0.2)

ax1.set_xlabel("Generation")
ax1.set_ylabel("Percentage (%)", color="green")
ax2.set_ylabel("Number", color="black")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("Average % Matings/Meetings and Number of Meetings and Matings per Generation, 0.0-0.45")
plt.show()