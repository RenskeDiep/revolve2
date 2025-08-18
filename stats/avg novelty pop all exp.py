# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 15:26:28 2025

@author: rensk
"""

import json
import numpy as np
import matplotlib.pyplot as plt

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/child_pop_dict.json."
with open(filename, "r") as f:
    exp_stand = json.load(f)
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/child_pop_dict.json."
with open(filename, "r") as f:
    exp_0_045 = json.load(f)
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/child_pop_dict.json."
with open(filename, "r") as f:
    exp_045_065= json.load(f)
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/child_pop_dict.json."
with open(filename, "r") as f:
    exp_065_1 = json.load(f)

# Collect values from all experiments
avg_list_stand = []
avg_list_0_045 = []
avg_list_045_065 = []
avg_list_065_1 = []


for exp_data in exp_stand.values():
    avg_list_stand.append(exp_data["avg_values"])
    
for exp_data in exp_0_045.values():
    avg_list_0_045.append(exp_data["avg_values"])
    
for exp_data in exp_045_065.values():
    avg_list_045_065.append(exp_data["avg_values"])
    
for exp_data in exp_065_1.values():
    avg_list_065_1.append(exp_data["avg_values"])

# Convert to NumPy arrays
avg_array_stand = np.array(avg_list_stand, dtype=float)
avg_array_0_045 = np.array(avg_list_0_045, dtype=float)
avg_array_045_065 = np.array(avg_list_045_065, dtype=float)
avg_array_065_1 = np.array(avg_list_065_1, dtype=float)

# Calculate mean ± std across experiments
all_avg_stand = np.nanmean(avg_array_stand, axis=0)
all_avg_0_045 = np.nanmean(avg_array_0_045, axis=0)
all_avg_045_065 = np.nanmean(avg_array_045_065, axis=0)
all_avg_065_1 = np.nanmean(avg_array_065_1, axis=0)

# Also compute std dev while ignoring NaN
#std_avg = np.nanstd(avg_array, axis=0)


# Compute means while ignoring NaN
all_avg_stand = np.nanmean(avg_array_stand, axis=0)
all_avg_0_045 = np.nanmean(avg_array_0_045, axis=0)
all_avg_045_065 = np.nanmean(avg_array_045_065, axis=0)
all_avg_065_1 = np.nanmean(avg_array_065_1, axis=0)


# Also compute std dev while ignoring NaN
#std_avg = np.nanstd(avg_array, axis=0)



gens = np.arange(len(all_avg_stand))

plt.plot(gens, all_avg_0_045, label="Avg. Novelty 0-0.45", color="blue")
plt.plot(gens, all_avg_045_065, label="Avg. Novelty 0.45-0.65", color="green")
plt.plot(gens, all_avg_065_1, label="Avg. Novelty 0.65-1.0", color="red")
plt.plot(gens, all_avg_stand, label="Avg. Novelty Standard Setup", color="purple")
#plt.fill_between(gens, all_avg - std_avg, all_avg + std_avg, color="blue", alpha=0.2)

plt.xlabel("Generation")
plt.ylabel("Novelty")
plt.title("Average Novelty of New Offspring in Population per Generation")
plt.legend()
plt.show()