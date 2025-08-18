# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:56:34 2025

@author: rensk
"""

import json
with open("C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 10/stats.json", "r") as f:
    data = json.load(f)
    

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


births = {}
deaths = {}

for generation in range(100):
    birth = 0
    death = 0
    for robot in data["robot_stats"]:
        initial_gen = data["robot_stats"][robot]["initial_generation"]
        max_gen = data["robot_stats"][robot]["final_generation"]
        if max_gen == None:
            max_gen = 99
        if generation == max_gen:
            death += 1
        if generation == initial_gen:
            birth += 1
    births[generation] = birth
    deaths[generation] = death


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/birth_deaths.json."
experiment_id = 10

if not os.path.exists(filename):
    with open(filename, "w+") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = {
    "births": births,
    "deaths": deaths
}

# Save back to the same file
with open(filename, "w+") as f:
    json.dump(all_data, f, indent=2)
