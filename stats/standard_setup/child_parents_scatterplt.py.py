# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 15:54:51 2025

@author: rensk
"""

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

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 1/child_parents.json."
with open(filename, "r") as f:
    run1 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 2/child_parents.json."
with open(filename, "r") as f:
    run2 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 3/child_parents.json."
with open(filename, "r") as f:
    run3 = json.load(f)
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 4/child_parents.json."
with open(filename, "r") as f:
    run4 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 5/child_parents.json."
with open(filename, "r") as f:
    run5 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 6/child_parents.json."
with open(filename, "r") as f:
    run6 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 7/child_parents.json."
with open(filename, "r") as f:
    run7 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 8/child_parents.json."
with open(filename, "r") as f:
    run8 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 9/child_parents.json."
with open(filename, "r") as f:
    run9 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/run 10/child_parents.json."
with open(filename, "r") as f:
    run10 = json.load(f)

parents1 = []
parents2 = []

runs = [run1, run2, run3, run4, run5, run6, run7, run8, run9, run10]
for run in runs:
    for robot in run:
        parent1 = run[robot][0][0]
        parent2 = run[robot][0][1]
        parents1.append(parent1)
        parents2.append(parent2)
        
plt.figure(figsize=(8, 6))
plt.scatter(parents1, parents2, color='blue', alpha=0.6)  # alpha makes points a bit transparent
plt.xlabel("Parents 1")
plt.ylabel("Parents 2")
plt.title("Similarity of New Offspring to Parents, Standard Setup")
plt.grid(True)
plt.show()