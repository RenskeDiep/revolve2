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


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/ages_per_gen.json."
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


combined = {}  # {"Gen 1": [all ages], "Gen 2": [all ages], ...}

for exp_data in all_runs.values():
    for gen, ages in exp_data.items():
        if gen not in combined:
            combined[gen] = []
        combined[gen].extend(ages)  # merge all ages

generations = combined.keys()
avg_ages = []
max_ages = []
std_ages = []

for gen in generations:
    ages = combined[gen]
    avg_ages.append(np.mean(ages))
    max_ages.append(np.max(ages))
    std_ages.append(np.std(ages))
#print(avg_ages)
# Convert generations to x-axis numbers for plotting
x = np.arange(len(generations))


# Max age line
plt.plot(x, max_ages, marker='o', color='red', label='Max Age')

# Shaded area for mean ± std

plt.fill_between(x, np.array(avg_ages) - np.array(std_ages),
                 np.array(avg_ages) + np.array(std_ages),
                 color='blue', alpha=0.2, label='Std Deviation')

plt.plot(x, avg_ages, color='blue', label='Average Age')

plt.xticks(x, generations)  # show generation names
plt.xlabel("Generation")
plt.ylabel("Age")
plt.ylim(0,100)
plt.xlim(0,100)
plt.title("Age Per Generation, 0.65 - 1.0")
plt.legend()
plt.xticks(   x[0::5], x[0::5])
plt.show()