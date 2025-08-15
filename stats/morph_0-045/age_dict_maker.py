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

ages_per_gen = defaultdict(list)
# Load JSON file
with open("C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 10/stats.json", "r") as f:
    data = json.load(f)
    
#print(data["robot_stats"])
for generation in range(100):
    for robot in data["robot_stats"]:
        initial_gen = data["robot_stats"][robot]["initial_generation"]
        max_gen = data["robot_stats"][robot]["final_generation"]
        if max_gen != None:
            if generation - initial_gen >= 0 and generation <= max_gen:
                age = generation - initial_gen
                ages_per_gen[generation].append(age)



filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/ages_per_gen.json."
experiment_id = 10

if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = ages_per_gen

# Save back to the same file
with open(filename, "w") as f:
    json.dump(all_data, f, indent=2)


