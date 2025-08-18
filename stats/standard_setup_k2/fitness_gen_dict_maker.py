# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:07:02 2025

@author: rensk
"""




import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

fitness_per_gen = defaultdict(list)
# Load JSON file
with open("C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/run 1/stats.json", "r") as f:
    data = json.load(f)
    
#print(data["robot_stats"])
for generation in range(100):
    for robot in data["robot_stats"]:
        initial_gen = data["robot_stats"][robot]["initial_generation"]
        max_gen = data["robot_stats"][robot]["final_generation"]
        if max_gen == None:
            max_gen = 99
        if generation > initial_gen and generation <= max_gen:
            fitness = data["robot_stats"][robot]["fitness"][str(generation)]
            if fitness != 0:
                fitness_per_gen[generation].append(fitness)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/fitness_per_gen.json."
experiment_id = 1

if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = fitness_per_gen

# Save back to the same file
with open(filename, "w") as f:
    json.dump(all_data, f, indent=2)


