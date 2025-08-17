# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 14:00:59 2025

@author: rensk
"""

# scatterplot similarity pop vs fitness of last gen. for each experiment, different color
# each run a dot. 
import json
import numpy as np
import matplotlib.pyplot as plt

# Load your data
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/similarity_dict.json."
with open(filename, "r") as f:
    sim_045_065 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/fitness_per_gen.json."
with open(filename, "r") as f:
    fitness_045_065 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/similarity_dict.json."
with open(filename, "r") as f:
    sim_0_045 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/fitness_per_gen.json."
with open(filename, "r") as f:
    fitness_0_045 = json.load(f)


last_gen = "99"
sim_fit_0_045 = {}
sim_fit_045_065 = {}

for exp in sim_045_065:  # loop over experiments
    for gen in sim_045_065[exp]:  # loop over generations
        if gen == last_gen:  # only look at the last generation
            sim_scores = sim_045_065[exp][gen].get("sim_scores", [])
            avg_sim= np.mean(sim_scores)
            fitness_scores = fitness_045_065[exp][gen]
            avg_fitness = np.mean(fitness_scores)
            print("Fitness: ", avg_fitness)
            print("sim: ", avg_sim)
            sim_fit_045_065[exp] = (avg_sim, avg_fitness)
            
for exp in sim_0_045:  # loop over experiments
    for gen in sim_0_045[exp]:  # loop over generations
        if gen == last_gen:  # only look at the last generation
            sim_scores = sim_0_045[exp][gen].get("sim_scores", [])
            avg_sim= np.mean(sim_scores)
            fitness_scores = fitness_0_045[exp][gen]
            avg_fitness = np.mean(fitness_scores)
            print("Fitness: ", avg_fitness)
            print("sim: ", avg_sim)
            sim_fit_0_045[exp] = (avg_sim, avg_fitness)


print(sim_fit_045_065)
print(sim_fit_0_045)


# Unpack data for first condition (0–0.45)
sims_0_045 = [v[0] for v in sim_fit_0_045.values()]
fitness_0_045 = [v[1] for v in sim_fit_0_045.values()]

# Unpack data for second condition (0.45–0.65)
sims_045_065 = [v[0] for v in sim_fit_045_065.values()]
fitness_045_065 = [v[1] for v in sim_fit_045_065.values()]

plt.figure(figsize=(8,6))

# Scatter for first group
plt.scatter(sims_0_045, fitness_0_045, color='blue', label='0.0–0.45')

# Scatter for second group
plt.scatter(sims_045_065, fitness_045_065, color='red', label='0.45–0.65')

z = np.polyfit(sims_0_045, fitness_0_045, 1)  # linear fit
p = np.poly1d(z)
x_range = np.linspace(min(sims_0_045), max(sims_0_045), 100)
plt.plot(x_range, p(x_range), color='blue', linestyle='--')

# --- Trendline for 0.45–0.65 ---
z = np.polyfit(sims_045_065, fitness_045_065, 1)  # linear fit
p = np.poly1d(z)
x_range = np.linspace(min(sims_045_065), max(sims_045_065), 100)
plt.plot(x_range, p(x_range), color='red', linestyle='--')

# Labels and title
plt.xlabel("Average Similarity")
plt.ylabel("Average Fitness")
plt.title("Fitness vs Similarity (last generation)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()