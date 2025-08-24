# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:25:52 2025

@author: rensk
"""

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


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/offspring_fitness_dict.json."
with open(filename, "r") as f:
    all_data = json.load(f)

plt.figure(figsize=(10, 6))

all_fitness = []
for exp in all_data:
    for robot_id, robot_data in all_data[exp].items():
        for offspring, fitness in robot_data:
            all_fitness.append(fitness)

# Define a threshold, e.g., remove top 5% fitness
threshold = np.percentile(all_fitness, 98)

x_vals = []
y_vals = []
for exp in all_data:
    for robot_id, robot_data in all_data[exp].items():
        for offspring, fitness in robot_data:
            if fitness <= threshold and fitness > 0:  # optional: remove outliers
                x_vals.append(offspring)
                y_vals.append(fitness)


# Loop through all experiments and robots
for exp in all_data:
    for robot_id, robot_data in all_data[exp].items():
        for offspring, fitness in robot_data:
            if fitness <= threshold:
                plt.scatter(offspring, fitness, color='blue', alpha=0.5)
coeffs = np.polyfit(x_vals, y_vals, deg=1)  # linear fit
print(coeffs)
from scipy.stats import linregress
result = linregress(x_vals, y_vals)

print("slope:", result.slope)
print("intercept:", result.intercept)
print("r-value:", result.rvalue)
print("p-value:", result.pvalue)
print("standard error:", result.stderr)
trendline = np.poly1d(coeffs)
plt.plot(sorted(x_vals), trendline(sorted(x_vals)), color='black', linewidth=1, label='Trendline degree 1')

plt.xlabel("Number of Offspring")
plt.ylabel("Fitness")
plt.title("Fitness per Offspring Count, 0.0-0.45")

plt.xlim(0, 110)   # x-axis range from 0 to 12
plt.ylim(0, 1.6) 
plt.grid(True)
#plt.xticks(range(max(x_vals)+1))  # show each number
plt.legend()
plt.show()