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

generations = []
matings = []
meetings = []

# Read the file
with open("C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 10/extra.txt", "r") as f:
    lines = f.readlines()

gen, mat, meet, count = None, None, None, None
for line in lines:
    line = line.strip()
    if line.startswith("Generation:"):
        gen = int(line.split(":")[1])
    elif line.startswith("Mating:"):
        mat = int(line.split(":")[1])
    elif line.startswith("Meeting:"):
        meet = int(line.split(":")[1])
    # When we have all three, save them
    if gen is not None and mat is not None and meet is not None:
        generations.append(gen)
        matings.append(mat)
        meetings.append(meet)
        gen, mat, meet, count = None, None, None, None  # reset for next block
        
        


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/mating_meeting_dict.json."
experiment_id = 10

if not os.path.exists(filename):
    with open(filename, "w") as f:
        json.dump({}, f)  # start with empty dict


with open(filename, "r") as f:
    all_data = json.load(f)


# Add current experiment
all_data[f"experiment_{experiment_id}"] = {
    "matings": matings,
    "meetings": meetings
}


# Save back to the same file
with open(filename, "w") as f:
    json.dump(all_data, f, indent=2)


