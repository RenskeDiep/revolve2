# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:16:01 2025

@author: rensk
"""

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

offspring_fitness = defaultdict(list)
offsprings = []
# Load JSON file
with open("C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/run 1/stats.json", "r") as f:
    data = json.load(f)
    

#print(data["robot_stats"])

for robot in data["robot_stats"]:
    for generation in range(100):
        total_fitness = 0
        initial_gen = data["robot_stats"][robot]["initial_generation"]
        max_gen = data["robot_stats"][robot]["final_generation"]
        if max_gen == None:
            max_gen = 99
        if generation > initial_gen and generation <= max_gen:
            fitness = data["robot_stats"][robot]["fitness"][str(generation)]
            total_fitness += fitness[0]
    offspring = data["robot_stats"][robot]["offspring_count"]
    offsprings.append(offspring)
    if fitness != 0:
        offspring_fitness[robot].append((offspring, fitness))


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/offspring_fitness_dict.json."
experiment_id = 1

if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = offspring_fitness

# Save back to the same file
with open(filename, "w") as f:
    json.dump(all_data, f, indent=2)
    

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/offsprings_dict.json."

if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = offsprings

# Save back to the same file
with open(filename, "w") as f:
    json.dump(all_data, f, indent=2)


