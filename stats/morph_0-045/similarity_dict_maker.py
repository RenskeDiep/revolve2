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

similarity_scores = defaultdict(lambda: {"sim_scores": []})

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 8/gen_to_robots.json."
with open(filename, "r") as f:
    gen_to_robot = json.load(f)
    gen_to_robot = gen_to_robot
    
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 8/uuid_to_measures.json."
with open(filename, "r") as f:
    robot_to_measures = json.load(f)
    
def similarity_score(v1, v2, normalize=True,
    method='euclidean'
)-> float:
    
    if normalize:
        max_vals = np.maximum(v1, v2)
        max_vals[max_vals == 0] = 1.0  # avoid divide-by-zero
        v1 /= max_vals
        v2 /= max_vals

    if method == 'euclidean':
        euclidean = np.linalg.norm(v1 - v2)
        max_dist = np.sqrt(len(v1))  # Max possible distance in normalized space
        #if (1 - (euclidean/max_dist)) > 1 or (1 - (euclidean/max_dist)) < 0:
            #print("TEST")
            #print(v1, v2)
            #print(euclidean, max_dist)
        return 1 - (euclidean / max_dist)
    else:
        raise ValueError(f"Unknown method: {method}")


for gen in gen_to_robot:
    population = gen_to_robot[gen]
    for r1, r2 in itertools.combinations(population, 2):
        v1 = robot_to_measures[r1][0][:6]
        v2 = robot_to_measures[r2][0][:6]
        sim_score = similarity_score(v1, v2)

        
        similarity_scores[gen]["sim_scores"].append(sim_score)
    similarity_scores[gen]["max"] = max(similarity_scores[gen]["sim_scores"])
    similarity_scores[gen]["avg"] = np.mean(similarity_scores[gen]["sim_scores"])
    similarity_scores[gen]["min"] = min(similarity_scores[gen]["sim_scores"])
            

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/similarity_dict.json."
experiment_id = 8

if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = similarity_scores


# Save back to the same file
with open(filename, "w") as f:
    json.dump(all_data, f, indent=2)
