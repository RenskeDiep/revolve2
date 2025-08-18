# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 22:15:05 2025

@author: rensk
"""

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


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/fitness_per_gen.json."
with open(filename, "r") as f:
    fitness_stand = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/fitness_per_gen.json."
with open(filename, "r") as f:
    fitness_065_1 = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/fitness_per_gen.json."
with open(filename, "r") as f:
    fitness_045_065 = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/fitness_per_gen.json."
with open(filename, "r") as f:
    fitness_0_045 = json.load(f)


combined_0_45 = {} 
combined_045_065 = {}  
combined_065_1 = {} 
combined_stand = {}   # {"Gen 1": [all fitness], "Gen 2": [all fitness], ...}

for exp_data in fitness_0_045.values():
    for gen, fitness in exp_data.items():
        if gen not in combined_0_45:
            combined_0_45[gen] = []
        combined_0_45[gen].extend(fitness)  # merge all fitn
        
for exp_data in fitness_045_065.values():
    for gen, fitness in exp_data.items():
        if gen not in combined_045_065:
            combined_045_065[gen] = []
        combined_045_065[gen].extend(fitness)  # merge all fitn
        
for exp_data in fitness_065_1.values():
    for gen, fitness in exp_data.items():
        if gen not in combined_065_1:
            combined_065_1[gen] = []
        combined_065_1[gen].extend(fitness)  # merge all fitn
        
for exp_data in fitness_stand.values():
    for gen, fitness in exp_data.items():
        if gen not in combined_stand:
            combined_stand[gen] = []
        combined_stand[gen].extend(fitness)  # merge all fitn

generations = combined_0_45.keys()
avg_fitness_0_045 = []
std_fitness_0_045 = []
avg_fitness_045_065 = []
std_fitness_045_065 = []
avg_fitness_065_1 = []
std_fitness_065_1 = []
avg_fitness_stand = []
std_fitness_stand = []

for gen in generations:
    fitness_0_045 = combined_0_45[gen]
    avg_fitness_0_045.append(np.mean(fitness_0_045))
    std_fitness_0_045.append(np.std(fitness_0_045))
    fitness_045_065 = combined_045_065[gen]
    avg_fitness_045_065.append(np.mean(fitness_045_065))
    std_fitness_045_065.append(np.std(fitness_045_065))
    fitness_065_1 = combined_065_1[gen]
    avg_fitness_065_1.append(np.mean(fitness_065_1))
    std_fitness_065_1.append(np.std(fitness_065_1))
    fitness_stand = combined_stand[gen]
    avg_fitness_stand.append(np.mean(fitness_stand))
    std_fitness_stand.append(np.std(fitness_stand))

# Convert generations to x-axis numbers for plotting
x = range(len(generations))

plt.figure(figsize=(8,6))

# Left axis: average fitness ± std
#plt.fill_between(x, np.array(avg_fitness_0_045) - np.array(std_fitness_0_045),
#                 np.array(avg_fitness_0_045) + np.array(std_fitness_0_045),
#                 color='blue', alpha=0.2, label='Std Deviation')
plt.plot(x, avg_fitness_0_045, color='blue', label='Avg. Fitness 0.0-0.45', linewidth=2)

# Left axis: average fitness ± std
#plt.fill_between(x, np.array(avg_fitness_045_065) - np.array(std_fitness_045_065),
#                 np.array(avg_fitness_045_065) + np.array(std_fitness_045_065),
#                 color='green', alpha=0.2, label='Std Deviation')
plt.plot(x, avg_fitness_045_065, color='green', label='Avg. Fitness 0.45-0.65', linewidth=2)

# Left axis: average fitness ± std
#plt.fill_between(x, np.array(avg_fitness_065_1) - np.array(std_fitness_065_1),
#                 np.array(avg_fitness_065_1) + np.array(std_fitness_065_1),
#                 color='red', alpha=0.2, label='Std Deviation')
plt.plot(x, avg_fitness_065_1, color='red', label='Avg. Fitness 0.65-1.0', linewidth=2)

# Left axis: average fitness ± std
#plt.fill_between(x, np.array(avg_fitness_stand) - np.array(std_fitness_stand),
#                 np.array(avg_fitness_stand) + np.array(std_fitness_stand),
#                 color='yellow', alpha=0.2, label='Std Deviation')
plt.plot(x, avg_fitness_stand, color='purple', label='Avg. Fitness Standard Setup', linewidth=2)



plt.xlabel("Generation")
plt.ylabel("Average Fitness ± Std", color='black')
plt.tick_params(axis='y', labelcolor='black')
plt.xticks( x[0::5], x[0::5]) # every 5th value )
#ax1.set_xticklabels([generations[i] for i in x[0::5]])
plt.ylim(bottom=0, top=2)

plt.legend()

plt.title("Average Fitness per Generation")
plt.show()





