import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load your data
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup/measures_dict.json."

with open(filename, "r") as f:
    all_experiments = json.load(f)

metrics = ["modules", "bricks", "branching_score", "num_limbs", "len_limbs", "coverage_score", "symmetry_score"]

# Initialize storage: metric -> gen -> list of values across experiments
metric_by_gen = {metric: defaultdict(list) for metric in metrics}

for exp_id, exp_data in all_experiments.items():
    for metric in metrics:
        if metric in exp_data:
            for gen_str, values in exp_data[metric].items():
                metric_by_gen[metric][int(gen_str)].append(np.mean(values))  # store average per experiment

# Compute overall average per generation
avg_per_gen = {metric: {} for metric in metrics}
std_per_gen = {metric: {} for metric in metrics}
for metric in metrics:
    for gen, values in metric_by_gen[metric].items():
        avg_per_gen[metric][gen] = np.mean(values)
        std_per_gen[metric][gen] = np.std(values)


# Sort generations
gens = sorted(avg_per_gen[metrics[0]].keys())

# Plotting
fig, ax1 = plt.subplots(figsize=(12,7))
ax2 = ax1.twinx()

for metric in metrics:
    if metric != "symmetry_score":
        y = [avg_per_gen[metric][g] for g in gens]
        y_std = [std_per_gen[metric][g] for g in gens]
        ax1.plot(gens, y, label=metric)
        ax1.fill_between(gens, np.array(y)-np.array(y_std), np.array(y)+np.array(y_std), alpha=0.2)
    if metric == "symmetry_score":
        y = [avg_per_gen[metric][g] for g in gens]
        y_std = [std_per_gen[metric][g] for g in gens]
        ax2.plot(gens, y, label=metric, color="pink")
        ax2.fill_between(gens, np.array(y)-np.array(y_std), np.array(y)+np.array(y_std),color="pink", alpha=0.2)

ax1.set_xlabel("Generation")
ax1.set_ylabel("Average metric ± std")
ax1.legend(loc="upper left")

ax2.set_ylabel("Symmetry ± std")

plt.xlabel("Generation")
plt.ylabel("Average value ± std")
plt.title("Average Morphology Measures per Generation across experiments, Standard Setup")
plt.legend()
plt.show()
