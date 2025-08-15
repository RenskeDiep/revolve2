# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 15:20:17 2025

@author: rensk
"""

import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 10/child_population.json."
with open(filename, "r") as f:
    data = json.load(f)

# novelty_by_gen[gen][robot_id] = list of similarities
novelty_by_gen = defaultdict(lambda: defaultdict(list))

for robot1, comparisons in data.items():
    for robot2, sim_score, generation in comparisons:
        # Store for robot1
        novelty_by_gen[generation][robot1].append(sim_score)
        # Also store for robot2
        novelty_by_gen[generation][robot2].append(sim_score)

# Determine the maximum generation so we can fill NaNs
max_gen = max(novelty_by_gen.keys())

novelty_per_gen_avg = {}
novelty_per_gen_max = {}
novelty_per_gen_min = {}

for gen in range(100):  # loop over *all* generations
    if gen in novelty_by_gen and novelty_by_gen[gen]:
        robot_novelties = []
        for sims in novelty_by_gen[gen].values():
            if sims:
                novelty = 1 - np.mean(sims)  # novelty = 1 - avg similarity
                robot_novelties.append(novelty)
        novelty_per_gen_avg[gen] = np.mean(robot_novelties) if robot_novelties else np.nan
        novelty_per_gen_max[gen] = np.max(robot_novelties) if robot_novelties else np.nan
        novelty_per_gen_min[gen] = np.min(robot_novelties) if robot_novelties else np.nan
    else:
        novelty_per_gen_avg[gen] = np.nan
        novelty_per_gen_max[gen] = np.nan
        novelty_per_gen_min[gen] = np.nan

# Sorted generations and aligned lists
gens_sorted = sorted(novelty_per_gen_avg.keys())
avg_values = [novelty_per_gen_avg[gen] for gen in gens_sorted]
max_values = [novelty_per_gen_max[gen] for gen in gens_sorted]
min_values = [novelty_per_gen_min[gen] for gen in gens_sorted]

# Save results
out_filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/child_pop_dict.json."
experiment_id = 10

if not os.path.exists(out_filename):
    with open(out_filename, "w+") as f:
        json.dump({}, f)  # start with empty dict

with open(out_filename, "r") as f:
    all_data = json.load(f)

all_data[f"experiment_{experiment_id}"] = {
    "avg_values": avg_values,
    "max_values": max_values,
    "min_values": min_values
}

with open(out_filename, "w+") as f:
    json.dump(all_data, f, indent=2)
