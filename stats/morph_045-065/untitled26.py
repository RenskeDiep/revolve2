# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:34:34 2025

@author: rensk
"""

import numpy as np
from scipy.stats import ttest_ind
import json


filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/measures_dict.json."

with open(filename, "r") as f:
    measures_0_045 = json.load(f)

filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_045-065/measures_dict.json."

with open(filename, "r") as f:
    measures_045_065 = json.load(f)
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/measures_dict.json."

with open(filename, "r") as f:
    measures_065_1 = json.load(f)
    
filename = "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/measures_dict.json."

with open(filename, "r") as f:
    measures_stand = json.load(f)


# ---- Step 1: Extract per-run averages for a feature ----
def extract_feature_means(experiments_dict, feature):
    run_means = []
    for run, data in experiments_dict.items():  # e.g. "experiment_1"
        if feature not in data:
            continue

        values = data[feature]

        # If it's a dict (keys like "0", "1", ...), flatten all lists
        if isinstance(values, dict):
            flat = []
            for v in values.values():
                flat.extend(v)   # extend with the list
        else:
            flat = values  # already a list

        flat = np.array(flat, dtype=float)  # ensure numeric
        run_means.append(np.nanmean(flat))  # mean per run

    return run_means
# ---- Step 2: Compare Standard vs other setups ----
def compare_setups(feature, standard_dict, setup_dicts):
    base = extract_feature_means(standard_dict, feature)
    results = {}
    for name, d in setup_dicts.items():
        other = extract_feature_means(d, feature)
        t, p = ttest_ind(base, other, equal_var=False)
        results[name] = {
            "mean_base": np.mean(base),
            "mean_other": np.mean(other),
            "t": t,
            "p": p,
            "significant": p < 0.05
        }
    return results

# ---- Step 3: Generate LaTeX table ----
def results_to_latex(all_results):
    header = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\begin{tabular}{|l|l|c|c|c|c|}\n"
        "\\hline\n"
        "Feature & Comparison & Mean (Exp) & Mean (Std) & t-value & p-value \\\\ \\hline\n"
    )
    rows = []
    for feature, results in all_results.items():
        for name, res in results.items():
            row = f"{feature} & {name} vs Std & {res['mean_other']:.3f} & {res['mean_base']:.3f} & {res['t']:.3f} & {res['p']:.4f} \\\\"
            rows.append(row)
    footer = "\\hline\n\\end{tabular}\n\\caption{Student’s t-tests for morphological features comparing each setup against Standard.}\n\\end{table}"
    return "\n".join([header] + rows + [footer])

# ---- Step 4: Run everything ----
features = ["modules", "bricks", "branching_score", "num_limbs", "len_limbs", "coverage_score"]

setups = {
    "0–0.45": measures_0_045,
    "0.45–0.65": measures_045_065,
    "0.65–1": measures_065_1
}

all_results = {}
for feature in features:
    all_results[feature] = compare_setups(feature, measures_stand, setups)

# Print LaTeX table
print(results_to_latex(all_results))
