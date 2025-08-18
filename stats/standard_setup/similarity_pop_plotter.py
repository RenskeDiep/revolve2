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


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/similarity_dict.json."
with open(filename, "r") as f:
    all_experiments = json.load(f)

# Get generations from one experiment
generations = sorted(all_experiments[next(iter(all_experiments))].keys(), key=int)
generations = [int(g) for g in generations]
x = np.arange(len(generations))

# Collect values per generation
avg_values_per_gen = []
max_values_per_gen = []
min_values_per_gen = []

for g in generations:
    avg_gen = [all_experiments[exp][str(g)]["avg"] for exp in all_experiments]
    max_gen = [all_experiments[exp][str(g)]["max"] for exp in all_experiments]
    min_gen = [all_experiments[exp][str(g)]["min"] for exp in all_experiments]
    avg_values_per_gen.append(avg_gen)
    max_values_per_gen.append(max_gen)
    min_values_per_gen.append(min_gen)

# Compute overall statistics
overall_avg = [np.mean(v) for v in avg_values_per_gen]
overall_max = [np.max(v) for v in max_values_per_gen]
overall_min = [np.min(v) for v in min_values_per_gen]
overall_std = [np.std(v) for v in avg_values_per_gen]

# Plot
plt.figure(figsize=(10,6))

for exp in all_experiments:
    exp_values = [all_experiments[exp][str(gen)]["avg"] for gen in generations]
    plt.plot(x, exp_values, color='gray', alpha=0.3)

plt.plot(generations, overall_avg, color='blue', label='Average')
plt.fill_between(generations, np.array(overall_avg)-np.array(overall_std),
                 np.array(overall_avg)+np.array(overall_std), color='blue', alpha=0.2, label='Std Dev')
plt.plot(generations, overall_max, color='red', linestyle='--', label='Max')
plt.plot(generations, overall_min, color='green', linestyle='--', label='Min')

plt.xlabel("Generation")
plt.ylabel("Diversity")
plt.title("Morphological Diversity per Generation, Standard Setup")
plt.legend()
plt.show()