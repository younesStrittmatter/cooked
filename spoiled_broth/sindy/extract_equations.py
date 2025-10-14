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
import numpy as np
import pandas as pd
import pysindy as ps
from pathlib import Path
from spoiled_broth.sindy.sindy_helpers import load_positions, compute_scaler, apply_scaler, extract_features_targets
import argparse


def extract_equations(map_identifier, base_path='/data/samuel_lozano/cooked/classic/v3.1/experiment', threshold=1e-4, poly_degree=2, fourier_n=0):
    experiment_base = Path(base_path) / f"map_{map_identifier}" / "simulations"
    output_dir = Path(base_path) / f"map_{map_identifier}" / "sindy_analysis" / f"threshold_{threshold}-poly_{poly_degree}-fourier_{fourier_n}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_pattern = str(experiment_base / "Training_*" / "checkpoint_*" / "simulation_*")
    simulation_dirs = glob.glob(sim_pattern)
    grouped = {}
    for sim_dir in simulation_dirs:
        parts = Path(sim_dir).parts
        try:
            training_idx = [i for i, p in enumerate(parts) if p.startswith('Training_')][0]
            checkpoint_idx = [i for i, p in enumerate(parts) if p.startswith('checkpoint_')][0]
            training_id = parts[training_idx].replace('Training_', '')
            checkpoint_number = parts[checkpoint_idx].replace('checkpoint_', '')
            key = (training_id, checkpoint_number)
            grouped.setdefault(key, []).append(sim_dir)
        except Exception:
            continue

    feature_names = ['pos_x_1','pos_y_1','vel_x_1','vel_y_1','pos_x_2','pos_y_2','vel_x_2','vel_y_2','rel_pos_x','rel_pos_y','rel_vel_x','rel_vel_y']
    target_names = ['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']

    for (training_id, checkpoint_number), sim_dirs in grouped.items():
        all_data = []
        for sim_dir in sim_dirs:
            pos1_path = Path(sim_dir) / "ai_rl_1_positions.csv"
            pos2_path = Path(sim_dir) / "ai_rl_2_positions.csv"
            if not pos1_path.exists() or not pos2_path.exists():
                continue
            pos1 = load_positions(pos1_path)
            pos2 = load_positions(pos2_path)
            features, targets = extract_features_targets(pos1, pos2)
            all_data.append((features, targets))
        if not all_data:
            continue
        Xs = [d for d, t in all_data]
        Xdots = [t for d, t in all_data]
        X_concat = np.vstack(Xs)
        Xdot_concat = np.vstack(Xdots)
        mu_x, sigma_x = compute_scaler(X_concat)
        mu_xdot, sigma_xdot = compute_scaler(Xdot_concat)
        Xs_scaled = [apply_scaler(Xi, mu_x, sigma_x) for Xi in Xs]
        Xdots_scaled = [apply_scaler(Xd, mu_xdot, sigma_xdot) for Xd in Xdots]
        # Fit SINDy
        poly_lib = ps.PolynomialLibrary(degree=poly_degree)
        feature_library = poly_lib
        if fourier_n and fourier_n > 0:
            fourier_lib = ps.FourierLibrary(n_frequencies=fourier_n)
            feature_library = poly_lib + fourier_lib
        optimizer = ps.STLSQ(threshold=threshold)
        sindy = ps.SINDy(optimizer=optimizer, feature_library=feature_library)
        sindy.fit(Xs_scaled, t=1, x_dot=Xdots_scaled)
        # Save equations
        equations = [sindy.equations(i) for i in range(len(target_names))]
        eq_path = output_dir / f'equations_{training_id}_{checkpoint_number}.txt'
        with open(eq_path, 'w') as f:
            for name, eq_str in zip(target_names, equations):
                f.write(f"{name}: {eq_str}\n")
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
            for i_term, term in enumerate(lib_names):
                for j_target in range(n_targets):
                    try:
                        coef_val = float(coef_matrix[i_term, j_target])
                    except Exception:
                        coef_val = float(coef_matrix[j_target, i_term]) if coef_matrix.shape[0] > j_target and coef_matrix.shape[1] > i_term else 0.0
                    target = target_names[j_target] if j_target < len(target_names) else f'target_{j_target}'
                    rows.append({'term': term, 'target': target, 'coefficient': coef_val, 'abs_coeff': abs(coef_val)})
            coef_df = pd.DataFrame(rows)
            coef_path = output_dir / f'coefficients_{training_id}_{checkpoint_number}.csv'
            coef_df.to_csv(coef_path, index=False)
        print(f"Saved equations and coefficients for Training_{training_id} / checkpoint_{checkpoint_number}")

def main():
    parser = argparse.ArgumentParser(description='Extract SINDy equations and coefficients')
    parser.add_argument('map_identifier', type=str)
    parser.add_argument('--base-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment')
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--poly-degree', type=int, default=2)
    parser.add_argument('--fourier-n', type=int, default=0)
    args = parser.parse_args()
    extract_equations(
        map_identifier=args.map_identifier,
        base_path=args.base_path,
        threshold=args.threshold,
        poly_degree=args.poly_degree,
        fourier_n=args.fourier_n
    )

if __name__ == "__main__":
    main()
