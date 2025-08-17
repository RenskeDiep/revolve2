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


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/fitness_per_gen.json."
with open(filename, "r") as f:
    all_runs = json.load(f)




plt.figure(figsize=(10, 6))

label_grey = True
for run_name, exp_data in all_runs.items():
    generations_run = exp_data.keys()
    avg_run = [
        np.mean([a for a in exp_data[gen]])
        for gen in generations_run
    ]
    if label_grey == True:  # add label only for the first gray line
        plt.plot(generations_run, avg_run, color='gray', alpha=0.3, label='Individual Runs')
    else:
        plt.plot(generations_run, avg_run, color='gray', alpha=0.3)
    label_grey = False


combined = {}  # {"Gen 1": [all fitness], "Gen 2": [all fitness], ...}

for exp_data in all_runs.values():
    for gen, fitness in exp_data.items():
        if gen not in combined:
            combined[gen] = []
        combined[gen].extend(fitness)  # merge all fitn

generations = combined.keys()
avg_fitness = []
max_fitness = []
std_fitness = []

for gen in generations:
    fitness = combined[gen]
    avg_fitness.append(np.mean(fitness))
    max_fitness.append(np.max(fitness))
    std_fitness.append(np.std(fitness))

# Convert generations to x-axis numbers for plotting
x = range(len(generations))

fig, ax1 = plt.subplots(figsize=(12,6))

# Left axis: average fitness ± std
ax1.fill_between(x, np.array(avg_fitness) - np.array(std_fitness),
                 np.array(avg_fitness) + np.array(std_fitness),
                 color='blue', alpha=0.2, label='Std Deviation')
ax1.plot(x, avg_fitness, color='blue', label='Average Fitness', linewidth=2)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Average Fitness ± Std", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks( x[0::5], x[0::5]) # every 5th value )
#ax1.set_xticklabels([generations[i] for i in x[0::5]])
ax1.set_ylim(bottom=0)

# Right axis: maximum fitness
ax2 = ax1.twinx()
ax2.plot(x, max_fitness, color='green', label='Maximum Fitness', linewidth=1)
ax2.set_ylabel("Maximum Fitness", color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(bottom=0)

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")

plt.title("Fitness per Generation, 0.65-1.0")
plt.show()