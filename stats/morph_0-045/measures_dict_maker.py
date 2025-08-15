# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 09:31:50 2025

@author: rensk
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools

num_modules = defaultdict(list)
num_bricks = defaultdict(list)
branching_score = defaultdict(list)
num_limbs = defaultdict(list)
len_limbs = defaultdict(list)
coverage_score = defaultdict(list)
symmetry_score = defaultdict(list)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 10/gen_to_robots.json."
with open(filename, "r") as f:
    gen_to_robot = json.load(f)
    gen_to_robot = gen_to_robot
    
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 10/uuid_to_measures.json."
with open(filename, "r") as f:
    robot_to_measures = json.load(f)
    


for gen in gen_to_robot:
    for robot in gen_to_robot[gen]:
        v1 = robot_to_measures[robot][0]
        modules, bricks, branching, limbs, len_of_limbs, coverage, symmetry = v1
        num_modules[gen].append(modules)
        num_bricks[gen].append(bricks)
        branching_score[gen].append(branching)
        num_limbs[gen].append(limbs)
        len_limbs[gen].append(len_of_limbs)
        coverage_score[gen].append(coverage)
        symmetry_score[gen].append(symmetry)
            

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/measures_dict.json."
experiment_id = 10

if not os.path.exists(filename):
    with open(filename, "w+") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = {
    "modules": num_modules,
    "bricks": num_bricks, 
    "branching_score": branching_score, 
    "num_limbs": num_limbs,
    "len_limbs": len_limbs,
    "coverage_score": coverage_score, 
    "symmetry_score":  symmetry_score
}

# Save back to the same file
with open(filename, "w+") as f:
    json.dump(all_data, f, indent=2)
