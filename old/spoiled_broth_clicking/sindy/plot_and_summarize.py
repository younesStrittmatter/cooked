#!/usr/bin/env python3
"""
Step 3: Plot figures and create summary JSON.
- Loads results, generates plots, writes summary_<training>_<checkpoint>.json and FINAL_RESULTS.json

Usage:
    nohup python3 plot_and_summarize.py --collab-dir /path/to/collaboration_jsons --output-path /path/to/summary.json > log_plot_and_summarize.out 2>&1 &

Example:
    nohup python3 plot_and_summarize.py --collab-dir /data/samuel_lozano/cooked/classic/v3.1/experiment/map_baseline_division_of_labor/sindy_analysis/threshold_0.0001-poly_2-fourier_0 --output-path /data/samuel_lozano/cooked/classic/v3.1/experiment/map_baseline_division_of_labor/sindy_analysis/threshold_0.0001-poly_2-fourier_0/summary_baseline_division_of_labor.json > log_plot_and_summarize.out 2>&1 &
"""
import os, glob, json
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import re

def parse_equations(eq_file_path):
    equations = {}
    with open(eq_file_path, 'r') as eqf:
        lines = eqf.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(accel_[xy]_[12]): (.+)", line)
            if m:
                acc = m.group(1)
                eq_str = m.group(2)
                equations[acc] = eq_str
    return equations

def plot_and_summarize(coeff_dir, data_path, summary_path):
    collab_files = glob.glob(os.path.join(coeff_dir, 'collaboration_*.json'))

    all_results = []
    mse_per_sim = {}
    r2_per_sim = {}
    true_all = {k: [] for k in ['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']}
    pred_all = {k: [] for k in ['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']}

    # Find equations file
    print(f"Looking for equations file in {coeff_dir}...")
    eq_files = glob.glob(os.path.join(coeff_dir, 'equations_*.txt'))
    if not eq_files:
        print('No equations file found!')
        return
    eq_file = eq_files[0]
    equations = parse_equations(eq_file)
    # Clean equations once after parsing
    for acc in equations:
        eq_clean = equations[acc].strip("'\"")
        expr = eq_clean.replace('^', '**')
        # Replace ' + ' with '+'
        expr = expr.replace(' + ', '+')
        # Replace '+-' with '-'
        expr = expr.replace('+-', '-')
        # Replace ' 1' with ' np.ones(len(x0))' to ensure proper dimensionality
        expr = expr.replace(' 1', ' np.ones(len(x0))')
        # Replace ' ' with '*' to ensure proper multiplication
        expr = expr.replace(' ', '*')
        # Replace 'exp(' with 'np.exp('
        expr = expr.replace('exp(', 'np.exp(')
        # Replace 'sin(' with 'np.sin(' and 'cos(' with 'np.cos('
        expr = expr.replace('sin(', 'np.sin(').replace('cos(', 'np.cos(')
        equations[acc] = expr
        print(f"Equation for {acc}: {expr}")

    # For each collaboration file, process all simulations for that training/checkpoint
    for collab_file in collab_files:
        # Extract training_id and checkpoint_number from filename
        fname = os.path.basename(collab_file)
        parts = fname.replace('collaboration_', '').replace('.json', '').split('_')
        if len(parts) < 3:
            print(f"Filename {fname} does not match expected pattern.")
            continue
        training_id = f'{parts[0]}_{parts[1]}'
        checkpoint_number = parts[2]

        # Find all simulation.csv files for this training/checkpoint
        sim_base = os.path.join(data_path, f'Training_{training_id}', f'checkpoint_{checkpoint_number}')
        for root, dirs, files in os.walk(sim_base):
            for file in files:
                if file == 'simulation.csv':
                    sim_csv = os.path.join(root, file)
                    # Extract simulation_id from path
                    sim_parts = os.path.relpath(sim_csv, sim_base).split(os.sep)
                    if len(sim_parts) < 2:
                        print(f"Unexpected simulation.csv path: {sim_csv}")
                        continue
                    simulation_id = sim_parts[0].replace('simulation_', '')
                    sim_df = pd.read_csv(sim_csv)
                    pred = {}
                    # Prepare normalized features for prediction
                    features = sim_df.iloc[:,1:13].values
                    for acc_idx, acc in enumerate(['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']):
                        eq = equations.get(acc, None)
                        if eq is None:
                            pred[acc] = np.zeros(len(sim_df))
                            continue
                        local_vars = {f'x{i}': features[:,i] for i in range(features.shape[1])}
                        try:
                            pred_norm = eval(eq, {'np': np}, local_vars)
                            pred[acc] = pred_norm
                        except Exception as e:
                            print(f'Error evaluating equation for {acc}: {e}')
                            pred[acc] = np.zeros(len(sim_df))
                        # Use standardized true values
                        true_standardized = sim_df[acc].values
                        true_all[acc].extend(true_standardized)
                        pred_all[acc].extend(pred[acc])
                    # Plot true vs predicted for this simulation (standardized values)
                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                    for i, acc in enumerate(['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']):
                        ax = axs[i//2, i%2]
                        ax.plot(sim_df['timestamp'], sim_df[acc].values, label='True (standardized)')
                        ax.plot(sim_df['timestamp'], pred[acc], label='Predicted (standardized)')
                        ax.set_title(acc + ' (standardized)')
                        ax.set_ylabel('Standardized acceleration')
                        ax.legend()
                    plt.tight_layout()
                    plot_name = f"true_vs_pred_{training_id}_{checkpoint_number}_{simulation_id}.png"
                    plt.savefig(os.path.join(coeff_dir, plot_name))
                    plt.close(fig)
                    mse_per_sim[plot_name] = {acc: mean_squared_error(sim_df[acc].values, pred[acc]) for acc in pred}
                    r2_per_sim[plot_name] = {acc: r2_score(sim_df[acc].values, pred[acc]) for acc in pred}
            if files:
                print(f"Finished processing simulations in {root}.")
                #break  # DEBUG: only first level
                

    # Aggregated and averaged plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, acc in enumerate(['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']):
        ax = axs[i//2, i%2]
        ax.plot(true_all[acc], label='True (aggregated)')
        ax.plot(pred_all[acc], label='Predicted (aggregated)')
        ax.set_title(acc)
        ax.set_ylabel('Standardized acceleration')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(coeff_dir, 'aggregated_true_vs_pred.png'))
    plt.close(fig)

    # Averaged plot (by time index)
    min_len = min(len(true_all[acc]) for acc in true_all)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, acc in enumerate(['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']):
        ax = axs[i//2, i%2]
        true_avg = np.mean(np.array(true_all[acc])[:min_len].reshape(-1, min_len), axis=0) if min_len > 0 else []
        pred_avg = np.mean(np.array(pred_all[acc])[:min_len].reshape(-1, min_len), axis=0) if min_len > 0 else []
        ax.plot(true_avg, label='True (avg)')
        ax.plot(pred_avg, label='Predicted (avg)')
        ax.set_title(acc)
        ax.set_ylabel('Standardized acceleration')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(coeff_dir, 'averaged_true_vs_pred.png'))
    plt.close(fig)

    # Compute aggregated and averaged MSE/R2, handling empty arrays
    mse_agg = {}
    r2_agg = {}
    for acc in true_all:
        if len(true_all[acc]) > 0 and len(pred_all[acc]) > 0:
            mse_agg[acc] = mean_squared_error(true_all[acc], pred_all[acc])
            r2_agg[acc] = r2_score(true_all[acc], pred_all[acc])
        else:
            mse_agg[acc] = None
            r2_agg[acc] = None

    mse_avg = {}
    r2_avg = {}
    for acc in true_all:
        if min_len > 0:
            true_avg = np.mean(np.array(true_all[acc])[:min_len].reshape(-1, min_len), axis=0) if min_len > 0 else []
            pred_avg = np.mean(np.array(pred_all[acc])[:min_len].reshape(-1, min_len), axis=0) if min_len > 0 else []
            if len(true_avg) > 0 and len(pred_avg) > 0:
                mse_avg[acc] = mean_squared_error(true_avg, pred_avg)
                r2_avg[acc] = r2_score(true_avg, pred_avg)
            else:
                mse_avg[acc] = None
                r2_avg[acc] = None
        else:
            mse_avg[acc] = None
            r2_avg[acc] = None

    # Only include averaged normalized metrics
    measures = list(all_results[0]['collaboration_measures'].keys()) if all_results else []
    summary = {
        'mse_per_simulation': mse_per_sim,
        'r2_per_simulation': r2_per_sim,
        'mse_aggregated': mse_agg,
        'r2_aggregated': r2_agg,
        'mse_averaged': mse_avg,
        'r2_averaged': r2_avg
    }
    for m in measures:
        vals = [r['collaboration_measures'][m]['normalized'] for r in all_results if m in r['collaboration_measures']]
        summary[f'avg_{m}_normalized'] = float(np.mean(vals)) if vals else None
    with open(summary_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved summary to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot and summarize collaboration results')
    parser.add_argument('map_identifier', type=str)
    parser.add_argument('--base-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment')
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--poly-degree', type=int, default=2)
    parser.add_argument('--fourier-n', type=int, default=0)
    args = parser.parse_args()

    # Define paths
    coeff_dir = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / f"threshold_{args.threshold}-poly_{args.poly_degree}-fourier_{args.fourier_n}"
    coeff_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / 'standardized_data'
    data_path.mkdir(parents=True, exist_ok=True)
    summary_path = Path(coeff_dir) / f'summary_{args.map_identifier}.json'
    summary_path.mkdir(parents=True, exist_ok=True)

    plot_and_summarize(
        coeff_dir=coeff_dir,
        data_path=data_path,
        summary_path=summary_path,
    )

if __name__ == "__main__":
    main()
