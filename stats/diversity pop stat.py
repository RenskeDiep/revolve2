# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 14:08:19 2025

@author: rensk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 11:37:35 2025

@author: rensk
"""

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
from numpy import trapz
from sklearn.metrics import auc
from scipy import stats

from scipy.stats import ttest_ind
import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

last_gen = "99"



filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/similarity_dict.json."
with open(filename, "r") as f:
    fitness_stand = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/similarity_dict.json."
with open(filename, "r") as f:
    fitness_065_1 = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/similarity_dict.json."
with open(filename, "r") as f:
    fitness_045_065 = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/similarity_dict.json."
with open(filename, "r") as f:
    fitness_0_045 = json.load(f)

from scipy.stats import ttest_ind
import numpy as np


# Helper function to get statistics per experiment
def get_fitness_stats(fitness_dict, last_gen):
    avg_all_gens = []
    best_all_gens = []
    avg_last_gen = []
    
    for exp in fitness_dict:
        all_gen_values = []
        for gen in fitness_dict[exp]:
            sim_scores = fitness_dict[exp][gen]["sim_scores"]
            gen_values = [1 - s for s in sim_scores]
            all_gen_values.extend(gen_values)  # combine all generations
            if gen == last_gen:
                avg_last_gen.append(np.mean(gen_values))
        if all_gen_values:
            avg_all_gens.append(np.mean(all_gen_values))
            best_all_gens.append(np.max(all_gen_values))
            
    return avg_all_gens, best_all_gens, avg_last_gen

last_gen = "99"

# Compute statistics
avg_0_045, best_0_045, final_avg_0_045 = get_fitness_stats(fitness_0_045, last_gen)
avg_045_065, best_045_065, final_avg_045_065 = get_fitness_stats(fitness_045_065, last_gen)
avg_065_1, best_065_1, final_avg_065_1 = get_fitness_stats(fitness_065_1, last_gen)
avg_stand, best_stand, final_avg_stand = get_fitness_stats(fitness_stand, last_gen)

# Compare each setup to standard using Student's t-test
setups = {
    "0-0.45": (avg_0_045, best_0_045, final_avg_0_045),
    "0.45-0.65": (avg_045_065, best_045_065, final_avg_045_065),
    "0.65-1": (avg_065_1, best_065_1, final_avg_065_1)
}

for name, (avg_fit, best_fit, final_fit) in setups.items():
    t_avg, p_avg = ttest_ind(avg_fit, avg_stand)
    t_best, p_best = ttest_ind(best_fit, best_stand)
    t_final, p_final = ttest_ind(final_fit, final_avg_stand)
    
    print(f"{name} vs standard:")
    print(f"  Avg fitness all gens: t={t_avg:.3f}, p={p_avg:.3f}")
    print(f"  Best fitness all gens: t={t_best:.3f}, p={p_best:.3f}")
    print(f"  Avg fitness last gen: t={t_final:.3f}, p={p_final:.3f}\n")
