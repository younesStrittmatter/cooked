#!/usr/bin/env python3
"""
Analysis script for competition reinforcement learning experiments.

This script analyzes training results from two-agent competitive experiments,
generating comprehensive visualizations and statistics.

Usage:
nohup python generate_figures_competition.py > generate_figures_competition.log 2>&1 &

Example:
nohup python generate_figures_competition.py > generate_figures_competition.log 2>&1 &
"""

# USE: nohup python analysis_classic.py <intent_version> <map_nr> > analysis_classic.log &

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# GRAPH SETTINGS
# Preserve math fonts and apply a larger global style similar to the user's request
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 30,
    "axes.labelsize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
    "figure.titlesize": 34,
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "legend.title_fontsize": 26,
    "lines.linewidth": 2,
})


# Human-readable names for attitude (alpha_beta) pairs used in the data
dict_attitudes = {
    "1.0_0.0": "Individualistic agent",
    "0.707_0.707": "Cooperative agent",
    "0.0_1.0": "Altruistic agent",
    "-0.707_0.707": "Sacrificial agent",
    "-1.0_0.0": "Martyrial agent",
    "-0.707_-0.707": "Destructive agent",
    "0.0_-1.0": "Spiteful agent",
    "0.707_-0.707": "Competitive agent",
}


def gaussian_smooth(y, sigma=1.0):
    """Simple Gaussian smoothing using a small kernel created with numpy.

    This avoids adding scipy as a dependency. The returned array has the
    same length as input and is convolved with a normalized Gaussian kernel.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    # radius covers +/- 3 sigma
    radius = int(max(1, np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    return np.convolve(y, kernel, mode='same')

intent_version = 'v3.1'
map_nr = 'simple_kitchen_competition'
smoothing_factor = 40

# Directorios base
#local = '/mnt/lustre/home/samuloza'
local = ''
#local = 'C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'

if intent_version is not None:
    raw_dir = f"{local}/data/samuel_lozano/cooked/competition/{intent_version}/map_{map_nr}"
else:
    raw_dir = f"{local}/data/samuel_lozano/cooked/competition/map_{map_nr}"

base_dirs = {
    "Competitive": f"{raw_dir}/competitive",
    "Cooperative": f"{raw_dir}/cooperative"
}

# Crear directorios base si no existen
for dir_path in base_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

output_path = f"{raw_dir}/training_results.csv"
figures_dir = f"{raw_dir}/paper_figures/"

# Crear también el directorio para las figuras si no existe
os.makedirs(figures_dir, exist_ok=True)

# Leer el CSV especificando los tipos de datos
dtype_dict = {
    "timestamp": str,
    "game_type": int,
    "alpha_1": float,
    "beta_1": float,
    "alpha_2": float,
    "beta_2": float
}

df = pd.read_csv(output_path, dtype=dtype_dict, low_memory=False)
for col in df.columns[6:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Crear una columna identificadora de combinación de coeficientes
df = df.sort_values(by=["alpha_1", "alpha_2"], ascending=[False, False])
df["attitude_key"] = df.apply(lambda row: f"{row['alpha_1']}_{row['beta_1']}_{row['alpha_2']}_{row['beta_2']}", axis=1)
df["pure_reward_total"] = df["pure_reward_ai_rl_1"] + df["pure_reward_ai_rl_2"]

df["total_deliveries"] = np.where(
    df["game_type"] == 1,
    df["total_deliveries_ai_rl_1"],
    df["total_deliveries_ai_rl_2"] + df["total_deliveries_ai_rl_2"]
)

# Filtrar todas las combinaciones únicas
unique_attitudes = df["attitude_key"].unique()
unique_lr = df["lr"].unique()
unique_game_type = df["game_type"].unique()

N = smoothing_factor
smoothed_figures_dir = os.path.join(figures_dir, f"smoothed_{N}")
os.makedirs(smoothed_figures_dir, exist_ok=True)

# Tunable sigma for smoothing the std used in the shaded shadows.
# Make this relatively large to produce much smoother shadows across epoch blocks.
# You can tweak this value (e.g. 3, 5, 10) to control smoothness.
shadow_sigma = 50.0
base_width = 0.25
max_growth = 1.75

# Also create combined plots showing both agents together for each attitude_key
for attitude in unique_attitudes:
    subset = df[df["attitude_key"] == attitude]

    att_parts = attitude.split('_')
    att1_title = f"{att_parts[0]}_{att_parts[1]}"
    att2_title = f"{att_parts[2]}_{att_parts[3]}"

    for game_type in unique_game_type:
        for lr in unique_lr:
            filtered_subset = subset[(subset["game_type"] == game_type) & (subset["lr"] == lr)].copy()
            filtered_subset["epoch_block"] = (filtered_subset["epoch"] // N)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Agent 1 delivered only
            middle_epochs = filtered_subset.groupby("epoch_block")["epoch"].median()

            own_grp_1 = filtered_subset.groupby("epoch_block")["delivered_own_ai_rl_1"].agg(["mean", "std"]).fillna(0)
            other_grp_1 = filtered_subset.groupby("epoch_block")["delivered_other_ai_rl_1"].agg(["mean", "std"]).fillna(0)

            x = middle_epochs.values
            # Colors: use blue/green for 'Own Salads' and orange/red for 'Other's Salads'
            own_color = "#1ABC9C"   # teal (blue/green)
            other_color = "#E67E22" # orange

            own_mean = own_grp_1["mean"].values
            own_std = own_grp_1["std"].values
            other_mean = other_grp_1["mean"].values
            other_std = other_grp_1["std"].values

            # Smooth the mean traces and create a smoothly growing shadow width
            smooth_mean_sigma = 5.0
            smoothed_own_mean = gaussian_smooth(own_mean, sigma=smooth_mean_sigma)

            growth_factor = np.linspace(0.0, max_growth, own_mean.size)
            shadow_width = gaussian_smooth(growth_factor, sigma=shadow_sigma)
            # Make shadow proportional to the (smoothed) mean value while keeping a minimum width
            mean_scale = np.abs(smoothed_own_mean)
            max_m = mean_scale.max() if mean_scale.max() > 0 else 1.0
            mean_factor = mean_scale / max_m
            shadow_width = shadow_width * (0.2 + 0.8 * mean_factor) + base_width

            # Plot the smoothed mean and proportional shadow
            ax1.plot(x, own_mean, label="Own salads", color=own_color)
            ax1.fill_between(x, smoothed_own_mean - shadow_width, smoothed_own_mean + shadow_width, color=own_color, alpha=0.25)
            
            
            smoothed_other_mean = gaussian_smooth(other_mean, sigma=smooth_mean_sigma)
            # Reuse same growth-based shadow but ensure matching length
            growth_factor_o = np.linspace(0.0, max_growth, other_mean.size)
            shadow_width_o = gaussian_smooth(growth_factor_o, sigma=shadow_sigma)
            mean_scale_o = np.abs(smoothed_other_mean)
            max_mo = mean_scale_o.max() if mean_scale_o.max() > 0 else 1.0
            mean_factor_o = mean_scale_o / max_mo
            shadow_width_o = shadow_width_o * (0.2 + 0.8 * mean_factor_o) + base_width

            ax1.plot(x, other_mean, label="Other's salads", color=other_color)
            ax1.fill_between(x, smoothed_other_mean - shadow_width_o, smoothed_other_mean + shadow_width_o, color=other_color, alpha=0.25)
            
            # Use rcParams for consistent label/title sizes and show human-readable attitude name
            name1 = dict_attitudes.get(att1_title, att1_title)
            ax1.set_ylabel("Number of salads delivered", fontsize=plt.rcParams.get('axes.labelsize'))
            ax1.set_title(f"{name1}", fontsize=plt.rcParams.get('axes.titlesize'))
            ax1.set_ylim(0, 6)
            ax1.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')

            # Agent 2 delivered only
            own_grp_2 = filtered_subset.groupby("epoch_block")["delivered_own_ai_rl_2"].agg(["mean", "std"]).fillna(0)
            other_grp_2 = filtered_subset.groupby("epoch_block")["delivered_other_ai_rl_2"].agg(["mean", "std"]).fillna(0)

            own_mean_2 = own_grp_2["mean"].values
            own_std_2 = own_grp_2["std"].values
            other_mean_2 = other_grp_2["mean"].values
            other_std_2 = other_grp_2["std"].values

            # Smooth means for agent 2 and use growth-based shadow width

            smooth_mean_sigma = 5.0
            smoothed_own_mean_2 = gaussian_smooth(own_mean_2, sigma=smooth_mean_sigma)

            growth_factor_2 = np.linspace(0.0, max_growth, own_mean_2.size)
            shadow_width_2 = gaussian_smooth(growth_factor_2, sigma=shadow_sigma)
            mean_scale_2 = np.abs(smoothed_own_mean_2)
            max_m2 = mean_scale_2.max() if mean_scale_2.max() > 0 else 1.0
            mean_factor2 = mean_scale_2 / max_m2
            shadow_width_2 = shadow_width_2 * (0.2 + 0.8 * mean_factor2) + base_width

            ax2.plot(x, own_mean_2, label="Own salads", color=own_color)
            ax2.fill_between(x, smoothed_own_mean_2 - shadow_width_2, smoothed_own_mean_2 + shadow_width_2, color=own_color, alpha=0.25)

            smoothed_other_mean_2 = gaussian_smooth(other_mean_2, sigma=smooth_mean_sigma)
            growth_factor_2o = np.linspace(0.0, max_growth, other_mean_2.size)
            shadow_width_2o = gaussian_smooth(growth_factor_2o, sigma=shadow_sigma)
            mean_scale_2o = np.abs(smoothed_other_mean_2)
            max_m2o = mean_scale_2o.max() if mean_scale_2o.max() > 0 else 1.0
            mean_factor2o = mean_scale_2o / max_m2o
            shadow_width_2o = shadow_width_2o * (0.2 + 0.8 * mean_factor2o) + base_width

            ax2.plot(x, other_mean_2, label="Other's salads", color=other_color)
            ax2.fill_between(x, smoothed_other_mean_2 - shadow_width_2o, smoothed_other_mean_2 + shadow_width_2o, color=other_color, alpha=0.25)
            
            name2 = dict_attitudes.get(att2_title, att2_title)
            ax2.set_xlabel("Epoch", fontsize=plt.rcParams.get('axes.labelsize'))
            ax2.set_ylabel("Number of salads delivered", fontsize=plt.rcParams.get('axes.labelsize'))
            ax2.set_title(f"{name2}", fontsize=plt.rcParams.get('axes.titlesize'))
            ax2.set_ylim(0, 6)
            ax2.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')

            plt.tight_layout()

            sanitized_attitude = attitude.replace('.', 'p')
            filename_combined = f"delivered_own_vs_other_combined_g{game_type}_lr{str(lr).replace('.', 'p')}_attitude_{sanitized_attitude}_smoothed_{N}.png"
            filepath_combined = os.path.join(smoothed_figures_dir, filename_combined)
            plt.savefig(filepath_combined)
            plt.close()

print("Combined figures created using only delivered_own and delivered_other per agent.")
