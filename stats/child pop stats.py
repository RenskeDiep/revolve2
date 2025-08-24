# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 15:22:09 2025

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



filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/child_pop_dict.json."
with open(filename, "r") as f:
    fitness_stand = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/child_pop_dict.json."
with open(filename, "r") as f:
    fitness_065_1 = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/child_pop_dict.json."
with open(filename, "r") as f:
    fitness_045_065 = json.load(f)


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/child_pop_dict.json."
with open(filename, "r") as f:
    fitness_0_045 = json.load(f)

import numpy as np
from scipy.stats import ttest_ind

def extract_run_means(setup_dict):
    """Take dict with experiment_1, experiment_2... and return mean novelty per run."""
    run_means = []
    for run in setup_dict.values():   # each run like experiment_1
        values = np.array(run["avg_values"], dtype=float)
        values = values[~np.isnan(values)]  # remove NaN
        if len(values) > 0:
            run_means.append(np.mean(values))
    return run_means

# Example: four setups (replace with your actual dicts)
setup_0045 = extract_run_means(fitness_0_045)  # 0–0.45
setup_045065 = extract_run_means(fitness_045_065)  # 0.45–0.65
setup_0651 = extract_run_means(fitness_065_1)  # 0.65–1
setup_standard = extract_run_means(fitness_stand)  # baseline

# T-tests vs standard
results = []
for setup, label in zip([setup_0045, setup_045065, setup_0651],
                        ["0–0.45", "0.45–0.65", "0.65–1"]):
    t, p = ttest_ind(setup, setup_standard, equal_var=False)
    results.append((label, np.mean(setup), np.mean(setup_standard), t, p))

# Bonferroni correction for 3 comparisons
alpha = 0.05
print(f"Corrected alpha: {alpha:.3f}\n")

for r in results:
    print(f"{r[0]} vs standard: mean={r[1]:.3f} vs {r[2]:.3f}, "
          f"t={r[3]:.3f}, p={r[4]:.4f}, significant={r[4] < alpha}")
