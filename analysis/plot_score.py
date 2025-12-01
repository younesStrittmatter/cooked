import os
import json
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import ttest_ind

# ===============================
#        CONFIG
# ===============================

base_path = "bundles"

conditions = {
    "nc_bl_mixed": 'baseline_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(0.4)]_[p(5, 2)_cs(0.2)_ws(1.0)]_ability_hints_large_map',
    "nc_bl_superstar": 'baseline_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(1.0)]_[p(5, 2)_cs(1.0)_ws(1.0)]_ability_hints_large_map',
    "nc_encouraged_mixed": 'encouraged_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(0.4)]_[p(5, 2)_cs(0.2)_ws(1.0)]_ability_hints_large_map',
    "nc_encouraged_superstar": 'encouraged_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(1.0)]_[p(5, 2)_cs(1.0)_ws(1.0)]_ability_hints_large_map',
    "c_encouraged_mixed": 'encouraged_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(0.4)]_[p(5, 2)_cs(0.2)_ws(1.0)]_ability_hints_large_map_with_collision',
    "c_encouraged_superstar": 'encouraged_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(1.0)]_[p(5, 2)_cs(1.0)_ws(1.0)]_ability_hints_large_map_with_collision',
    "c_bl_superstar": "baseline_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(1.0)]_[p(5, 2)_cs(1.0)_ws(1.0)]_ability_hints_large_map_with_collision",
    "c_bl_mixed": "baseline_division_of_labor_large_[p(1, 2)_cs(1.0)_ws(0.4)]_[p(5, 2)_cs(0.2)_ws(1.0)]_ability_hints_large_map_with_collision",
}

folders = {k: os.path.join(base_path, v) for k, v in conditions.items()}


# ===============================
#      LOAD SCORES
# ===============================

def load_scores(folder_path):
    scores = []
    if not os.path.isdir(folder_path):
        return scores
    for sub in os.listdir(folder_path):
        meta = os.path.join(folder_path, sub, "meta.json")
        if os.path.isfile(meta):
            try:
                with open(meta) as f:
                    j = json.load(f)
                if "final_score" in j:
                    scores.append(j["final_score"])
            except:
                pass
    return scores


all_scores = {name: load_scores(path) for name, path in folders.items()}


# ===============================
#      PRINT MEANS + T-TESTS
# ===============================

print("\n=== GROUP MEANS ===")
for name, vals in all_scores.items():
    if len(vals) == 0:
        print(f"{name:25s} NO DATA FOUND")
    else:
        print(f"{name:25s} mean = {sum(vals) / len(vals):.2f}   n={len(vals)}")

print("\n=== ALL PAIRWISE T-TESTS ===")
for a, b in itertools.combinations(all_scores.keys(), 2):
    if len(all_scores[a]) > 1 and len(all_scores[b]) > 1:
        t, p = ttest_ind(all_scores[a], all_scores[b], equal_var=False)
        print(f"{a:25s} vs {b:25s}   t={t:.3f}, p={p:.5f}")
    else:
        print(f"{a:25s} vs {b:25s}   NOT ENOUGH DATA")


# ===============================
#      PLOTTING (FULLY DYNAMIC)
# ===============================

# Sort keys for consistent ordering (customize if needed)
ordered_keys = sorted(all_scores.keys())

data = [all_scores[k] for k in ordered_keys]
labels = ordered_keys

plt.figure(figsize=(12, 6))

# -------------------------------
#           VIOLINS
# -------------------------------
parts = plt.violinplot(data, showmeans=False, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor('#8da0cb')
    pc.set_alpha(0.75)
    pc.set_edgecolor("black")

# -------------------------------
#   HORIZONTAL STACKED DOTS
# -------------------------------
dot_spacing = 0.03
dot_size = 40

for xi, vals in enumerate(data, start=1):
    counts = Counter(vals)
    for score, count in sorted(counts.items()):
        offsets = [(i - (count - 1) / 2) * dot_spacing for i in range(count)]
        x_positions = [xi + off for off in offsets]
        plt.scatter(
            x_positions,
            [score] * count,
            color="gray",
            edgecolor="black",
            s=dot_size,
            zorder=3,
            linewidths=0.3,
        )

# -------------------------------
#       MEAN MARKER
# -------------------------------
for xi, vals in enumerate(data, start=1):
    if len(vals) > 0:
        plt.scatter(
            xi,
            sum(vals) / len(vals),
            color="magenta",
            marker="D",
            s=130,
            edgecolor="black",
            zorder=4,
        )

plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
plt.ylabel("Final Score")
plt.title("Final Score Distributions Across All Conditions")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
