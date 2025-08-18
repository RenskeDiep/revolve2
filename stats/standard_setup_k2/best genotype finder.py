# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:36:27 2025

@author: rensk
"""
import json

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/run 10/stats.json."
with open(filename, "r") as f:
    data = json.load(f)

best_uuid = None
best_fitness = float("-inf")
best_genotype = None

for uuid, stats in data["robot_stats"].items():  # assuming your big dict is called `data`
    max_fit = max(stats["fitness"].values())
    if max_fit < 14 and max_fit > best_fitness:
        best_fitness = max_fit
        best_uuid = uuid
        best_genotype = stats["genotype"]

print("Best robot UUID:", best_uuid)
print("Best fitness:", best_fitness)
print("best genotype: ", best_genotype)

# `best_genotype` now holds the actual Genotype object