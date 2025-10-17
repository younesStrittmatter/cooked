#!/usr/bin/env python3
"""
Step 1: Extract SINDy equations and coefficients from simulation data.
- Loads positions, computes features/targets, fits SINDy.
- Saves equations (txt) and coefficients (csv) per checkpoint.

Usage:
    nohup python3 extract_equations.py <map_identifier> [--base-path /path/to/experiment] [--threshold 1e-4] [--poly-degree 2] [--fourier-n 0] > log_extract_equations.out 2>&1 &

    nohup python3 extract_equations.py baseline_division_of_labor > log_extract_equations.out 2>&1 &
"""
import glob
import json
import numpy as np
import pandas as pd
import pysindy as ps
from pathlib import Path
from spoiled_broth.sindy.sindy_helpers import load_positions, compute_scaler, apply_scaler, extract_features_targets
import argparse


def prepare_standardized_data(experiment_path, data_path):
    sim_pattern = str(experiment_path / "Training_*" / "checkpoint_*" / "simulation_*")
    simulation_dirs = glob.glob(sim_pattern)
    grouped = {}
    for sim_dir in simulation_dirs:
        parts = Path(sim_dir).parts
        try:
            training_idx = [i for i, p in enumerate(parts) if p.startswith('Training_')][0]
            checkpoint_idx = [i for i, p in enumerate(parts) if p.startswith('checkpoint_')][0]
            simulation_idx = [i for i, p in enumerate(parts) if p.startswith('simulation_')][0]
            training_id = parts[training_idx].replace('Training_', '')
            checkpoint_number = parts[checkpoint_idx].replace('checkpoint_', '')
            simulation_id = parts[simulation_idx].replace('simulation_', '')
            key = (training_id, checkpoint_number, simulation_id)
            grouped.setdefault(key, []).append(sim_dir)
        except Exception:
            continue

    # Gather all features/targets for normalization
    all_features = []
    all_targets = []
    all_data = []
    for (training_id, checkpoint_number, simulation_id), sim_dirs in grouped.items():
        for sim_dir in sim_dirs:
            pos1_path = Path(sim_dir) / "ai_rl_1_positions.csv"
            pos2_path = Path(sim_dir) / "ai_rl_2_positions.csv"
            if not pos1_path.exists() or not pos2_path.exists():
                continue
            pos1 = load_positions(pos1_path)
            pos2 = load_positions(pos2_path)
            features, targets = extract_features_targets(pos1, pos2)
            all_features.append(features)
            all_targets.append(targets)
            all_data.append((features, targets))

    # Standardization (mean/std)
    features_concat = np.vstack(all_features)
    targets_concat = np.vstack(all_targets)
    feature_mean = features_concat.mean(axis=0)
    feature_std = features_concat.std(axis=0)
    target_mean = targets_concat.mean(axis=0)
    target_std = targets_concat.std(axis=0)

    # Save normalization factors
    norm_dict = {
        'feature_mean': feature_mean.tolist(),
        'feature_std': feature_std.tolist(),
        'target_mean': target_mean.tolist(),
        'target_std': target_std.tolist(),
        'feature_names': ['pos_x_1','pos_y_1','vel_x_1','vel_y_1','pos_x_2','pos_y_2','vel_x_2','vel_y_2','rel_pos_x','rel_pos_y','rel_vel_x','rel_vel_y'],
        'target_names': ['accel_x_1','accel_y_1','accel_x_2','accel_y_2']
    }
    norm_path = Path(data_path) / 'normalization.json'
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(norm_path, 'w') as f:
        json.dump(norm_dict, f, indent=2)

    # Save normalized CSVs
    idx = 0
    for (training_id, checkpoint_number, simulation_id), sim_dirs in grouped.items():
        for sim_dir in sim_dirs:
            pos1_path = Path(sim_dir) / "ai_rl_1_positions.csv"
            pos2_path = Path(sim_dir) / "ai_rl_2_positions.csv"
            if not pos1_path.exists() or not pos2_path.exists():
                continue
            features, targets = all_features[idx], all_targets[idx]
            idx += 1
            n_steps = features.shape[0]
            features_norm = (features - feature_mean) / (feature_std + 1e-8)
            targets_norm = (targets - target_mean) / (target_std + 1e-8)
            sim_df = pd.DataFrame(
                np.hstack([
                    np.arange(n_steps).reshape(-1, 1),
                    features_norm,
                    targets_norm
                ]),
                columns=[
                    'timestamp',
                    'pos_x_1','pos_y_1','vel_x_1','vel_y_1',
                    'pos_x_2','pos_y_2','vel_x_2','vel_y_2',
                    'rel_pos_x','rel_pos_y','rel_vel_x','rel_vel_y',
                    'accel_x_1','accel_y_1','accel_x_2','accel_y_2'
                ]
            )
            sim_csv_dir = Path(data_path) / f"Training_{training_id}" / f"checkpoint_{checkpoint_number}" / f"simulation_{simulation_id}"
            sim_csv_dir.mkdir(parents=True, exist_ok=True)
            sim_csv_path = sim_csv_dir / "simulation.csv"
            sim_df.to_csv(sim_csv_path, index=False)

    print(f"Saved standardized CSVs under {data_path}")

    return str(data_path)

def extract_equations_from_standardized(data_path, coeff_dir, threshold=1e-4, poly_degree=2, fourier_n=0):
    """Read standardized CSVs from data_path and run SINDy per checkpoint, saving equations and coefficients to coeff_dir."""
    feature_names = ['pos_x_1','pos_y_1','vel_x_1','vel_y_1','pos_x_2','pos_y_2','vel_x_2','vel_y_2','rel_pos_x','rel_pos_y','rel_vel_x','rel_vel_y']
    target_names = ['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']

    sim_pattern = str(Path(data_path) / "Training_*" / "checkpoint_*" / "simulation_*" / "simulation.csv")
    sim_files = glob.glob(sim_pattern)
    grouped = {}
    for f in sim_files:
        parts = Path(f).parts
        try:
            training_idx = [i for i, p in enumerate(parts) if p.startswith('Training_')][0]
            checkpoint_idx = [i for i, p in enumerate(parts) if p.startswith('checkpoint_')][0]
            training_id = parts[training_idx].replace('Training_', '')
            checkpoint_number = parts[checkpoint_idx].replace('checkpoint_', '')
            key = (training_id, checkpoint_number)
            grouped.setdefault(key, []).append(f)
        except Exception:
            continue

    for (training_id, checkpoint_number), files in grouped.items():
        Xs = []
        Xdots = []
        for sim_csv in files:
            df = pd.read_csv(sim_csv)
            # Features are the standardized feature columns
            X = df[['pos_x_1','pos_y_1','vel_x_1','vel_y_1','pos_x_2','pos_y_2','vel_x_2','vel_y_2','rel_pos_x','rel_pos_y','rel_vel_x','rel_vel_y']].values
            Y = df[['accel_x_1','accel_y_1','accel_x_2','accel_y_2']].values
            Xs.append(X)
            Xdots.append(Y)

        # Fit SINDy on the standardized data (no further scaling)
        poly_lib = ps.PolynomialLibrary(degree=poly_degree)
        feature_library = poly_lib
        if fourier_n and fourier_n > 0:
            fourier_lib = ps.FourierLibrary(n_frequencies=fourier_n)
            feature_library = poly_lib + fourier_lib
        optimizer = ps.STLSQ(threshold=threshold)
        sindy = ps.SINDy(optimizer=optimizer, feature_library=feature_library)
        sindy.fit(Xs, t=1, x_dot=Xdots)

        # Save equations
        equations = sindy.equations()  # returns all equations once, cleanly
        for name, eq in zip(target_names, equations):
            print(f'Equation for {name}: {eq}')
        
        eq_path = Path(coeff_dir) / f'equations_{training_id}_{checkpoint_number}.txt'
        with open(eq_path, 'w') as f:
            for name, eq in zip(target_names, equations):
                f.write(f"{name}: {eq}\n")

        # Save coefficients
        try:
            lib_names = sindy.get_feature_names(feature_names)
        except Exception:
            lib_names = None
            if hasattr(sindy, 'feature_library') and hasattr(sindy.feature_library, 'get_feature_names'):
                try:
                    lib_names = sindy.feature_library.get_feature_names(feature_names)
                except Exception:
                    lib_names = None
        try:
            coef_matrix = sindy.coefficients()
        except Exception:
            coef_matrix = getattr(sindy, 'coef_', None)
        if coef_matrix is not None and coef_matrix.ndim == 2:
            n_rows, n_cols = coef_matrix.shape
            if n_rows == len(target_names) and n_cols > len(target_names):
                coef_matrix = coef_matrix.T
        if lib_names is not None and coef_matrix is not None:
            rows = []
            n_terms = len(lib_names)
            n_targets = coef_matrix.shape[1]
            eps = 1e-12
            for i_term, term in enumerate(lib_names):
                for j_target in range(n_targets):
                    try:
                        coef_val = float(coef_matrix[i_term, j_target])
                    except Exception:
                        coef_val = float(coef_matrix[j_target, i_term]) if coef_matrix.shape[0] > j_target and coef_matrix.shape[1] > i_term else 0.0
                    target = target_names[j_target] if j_target < len(target_names) else f'target_{j_target}'
                    abs_val = abs(coef_val)
                    is_nz = abs_val > eps
                    rows.append({'term': term, 'target': target, 'coefficient': coef_val, 'abs_coeff': abs_val, 'is_nonzero': bool(is_nz)})
            coef_df = pd.DataFrame(rows)
            coef_path = Path(coeff_dir) / f'coefficients_{training_id}_{checkpoint_number}.csv'
            coef_df.to_csv(coef_path, index=False)

            # Compute per-target statistics and save a checkpoint summary
            per_target = {}
            total_abs_all = float(coef_df['abs_coeff'].sum()) if not coef_df['abs_coeff'].empty else 0.0
            for t in coef_df['target'].unique():
                sub = coef_df[coef_df['target'] == t]
                total_terms = int(len(sub))
                nonzero_count = int(sub['is_nonzero'].sum())
                zero_count = int(total_terms - nonzero_count)
                sum_abs = float(sub['abs_coeff'].sum()) if not sub['abs_coeff'].empty else 0.0
                nonzero_sum_abs = float(sub.loc[sub['is_nonzero'], 'abs_coeff'].sum()) if not sub.loc[sub['is_nonzero'], 'abs_coeff'].empty else 0.0
                per_target[t] = {
                    'total_terms': total_terms,
                    'nonzero_count': nonzero_count,
                    'zero_count': zero_count,
                    'sum_abs_coeff': sum_abs,
                    'nonzero_sum_abs_coeff': nonzero_sum_abs,
                    'fraction_of_total_abs': (sum_abs / total_abs_all) if total_abs_all > 0 else None
                }
            summary = {
                'training_id': training_id,
                'checkpoint_number': checkpoint_number,
                'total_abs_all_terms': total_abs_all,
                'per_target': per_target
            }
            summary_path = Path(coeff_dir) / f'summary_coefficients.json'
            with open(summary_path, 'w') as sf:
                json.dump(summary, sf, indent=2)
        print(f"Saved equations and coefficients for Training_{training_id} / checkpoint_{checkpoint_number}")

def main():
    parser = argparse.ArgumentParser(description='Extract SINDy equations and coefficients')
    parser.add_argument('map_identifier', type=str)
    parser.add_argument('--base-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment')
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--poly-degree', type=int, default=2)
    parser.add_argument('--fourier-n', type=int, default=0)
    args = parser.parse_args()

    # compute directories
    experiment_path = Path(args.base_path) / f"map_{args.map_identifier}" / "simulations"
    coeff_dir = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / f"threshold_{args.threshold}-poly_{args.poly_degree}-fourier_{args.fourier_n}"
    coeff_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / 'standardized_data'
    data_path.mkdir(parents=True, exist_ok=True)

    # Step 1: prepare standardized CSVs (and normalization.json)
    prepare_standardized_data(experiment_path, data_path)

    # Step 2: run SINDy using standardized CSVs
    extract_equations_from_standardized(data_path, coeff_dir, threshold=args.threshold, poly_degree=args.poly_degree, fourier_n=args.fourier_n)

if __name__ == "__main__":
    main()
