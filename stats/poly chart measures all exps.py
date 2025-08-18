# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:27:12 2025

@author: rensk
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/measures_dict.json."

with open(filename, "r") as f:
    exp1 = json.load(f)

metrics = ["modules", "bricks", "branching_score", "num_limbs", "coverage_score", "len_limbs"]


all_experiment_means = defaultdict(list)

for metric in metrics:
    for experiment in exp1.keys():
        mean = np.mean(exp1[experiment][metric]['99'])
        all_experiment_means[metric].append(mean)
        
mean_list = [np.mean(values) for values in all_experiment_means.values()]

print(mean_list)
                     
#for exp_name, exp_data in exp1.items():
#    print(exp_name)
#    print(exp_data)
#    modules = exp_data["modules"]["99"]
#    modules = exp_data["modules"]["99"]
#    modules = exp_data["modules"]["99"]
#    modules = exp_data["modules"]["99"]
#    print(modules)
    # get the last generation key
    #last_gen_key = max(modules.keys(), key=lambda x: int(x))
    #last_gen = np.array(modules[last_gen_key])  # shape: (num_individuals, num_metrics)

    # average per metric across individuals
    #mean_per_metric = last_gen.mean(axis=0)
    #all_experiment_means.append(mean_per_metric)

# average across experiments
#final_mean = np.mean(all_experiment_means, axis=0)
#print(all_experiment_means)



# Example data: replace with your real measures
v1 = np.array([18.57909195402299, 4.615936781609196, 0.45912707991242474, 0.5656675871701733,  0.00548853478917157, 0.46861506234168504], dtype=float)
v2 = np.array([19.349885057471262, 5.2091954022988505, 0.4096617405582923, 0.5517575858049997,  0.005399045480874165, 0.4837875160580536], dtype=float)
v3 = np.array([19.96333333333333, 5.142183908045977, 0.4220925013683634, 0.5537325056807816,  0.004922449203894815, 0.4887857031594353], dtype=float)
v4 = np.array([23.402857142857144, 5.717142857142857, 0.28840136054421767, 0.4349234971377828,  0.0011168568323775532, 0.6187267310187479], dtype=float)

labels = [
    "Num Modules",
    "Num Bricks",
    "Branching",
    "Limbs",
    "Coverage",
    "Length of Limbs"
]

experiments = [v1, v2, v3, v4]
exp_names = ["Similarity 0-0.45", "Similarity 0.45-0.65", "Similarity 0.65-1", "Standard setup"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# --- Normalize across all experiments ---
all_data = np.vstack(experiments)
mins = [0, 0, 0, 0, 0, 0]
maxs = all_data.max(axis=0)
norm = (all_data - mins) / (maxs - mins)

# Radar setup
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

def close_loop(values):
    return np.concatenate([values, [values[0]]])

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each experiment
for i, (values, label, color) in enumerate(zip(norm, exp_names, colors)):
    vals = close_loop(values)
    ax.plot(angles, vals, linewidth=2, color=color, label=label)
    ax.fill(angles, vals, alpha=0.15, color=color)

# Add axis labels with real ranges
for i, (angle, lab) in enumerate(zip(angles[:-1], labels)):
    ax.text(
        angle,
        1.15,  # slightly outside the circle
        f"{lab}\n(min {mins[i]:.2f}, max {maxs[i]:.2f})",
        ha="center",
        va="center",
        fontsize=9,
    )

# Style
#ax.set_xticks([])  # remove default labels
#ax.set_yticks([])
ax.set_yticklabels([]) 
#ax.set_xticks([]) 
ax.set_xticklabels([]) 
ax.spines["polar"].set_visible(True)
ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

ax.set_title("Robot Morphology Measures", size=14, weight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.show()
