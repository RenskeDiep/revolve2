# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 15:26:28 2025

@author: rensk
"""

import json
import numpy as np
import matplotlib.pyplot as plt

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/child_pop_dict.json."
with open(filename, "r") as f:
    all_data = json.load(f)

# Collect values from all experiments
avg_list = []
max_list = []
min_list = []

for exp_data in all_data.values():
    avg_list.append(exp_data["avg_values"])
    max_list.append(exp_data["max_values"])
    min_list.append(exp_data["min_values"])

# Convert to NumPy arrays
avg_array = np.array(avg_list, dtype=float)
max_array = np.array(max_list, dtype=float)
min_array = np.array(min_list, dtype=float)

# Calculate mean ± std across experiments
all_avg = np.nanmean(avg_array, axis=0)
all_max = np.nanmean(max_array, axis=0)
all_min = np.nanmean(min_array, axis=0)

# Also compute std dev while ignoring NaN
std_avg = np.nanstd(avg_array, axis=0)


# Compute means while ignoring NaN
all_avg = np.nanmean(avg_array, axis=0)
all_max = np.nanmean(max_array, axis=0)
all_min = np.nanmean(min_array, axis=0)

# Also compute std dev while ignoring NaN
std_avg = np.nanstd(avg_array, axis=0)

# Example plot
import matplotlib.pyplot as plt

gens = np.arange(len(all_avg))

plt.plot(gens, all_avg, label="Average novelty", color="blue")
plt.fill_between(gens, all_avg - std_avg, all_avg + std_avg, color="blue", alpha=0.2)
plt.plot(gens, all_max, label="Max novelty", color="green")
plt.plot(gens, all_min, label="Min novelty", color="red")

plt.xlabel("Generation")
plt.ylabel("Novelty")
plt.title("Average Novelty of New Offspring in Population per Generation, 0.45-0.65")
plt.legend()
plt.show()