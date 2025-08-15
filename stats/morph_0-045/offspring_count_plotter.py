# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:25:34 2025

@author: rensk
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter



filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/offsprings_dict.json."
with open(filename, "r") as f:
    all_data = json.load(f)
    
# Collect all offspring numbers from all experiments
all_offspring = []
for exp_list in all_data.values():
    all_offspring.extend(exp_list)

# Count how many robots have each number of offspring
offspring_counts = Counter(all_offspring)

# Sort by offspring number
sorted_offspring = sorted(offspring_counts.items())
x_vals = [x for x, _ in sorted_offspring]
y_vals = [y for _, y in sorted_offspring]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(x_vals, y_vals, color='skyblue')
plt.xlabel("Number of Offspring")
plt.ylabel("Counts")
plt.title("Distribution of Offspring Counts, 0.0-0.45")
plt.xticks(range(max(x_vals)+1))  # show each number
plt.show()