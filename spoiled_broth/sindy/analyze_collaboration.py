#!/usr/bin/env python3
"""
Step 2: Analyze collaboration from coefficients CSVs.
- Loads coefficients CSV, computes 7 collaboration measures per checkpoint.
- Saves results as collaboration_<training>_<checkpoint>.json

Usage:
    nohup python3 analyze_collaboration.py --coeff-dir /path/to/coeffs > log_analyze_collaboration.out 2>&1 &
    
    nohup python3 analyze_collaboration.py --coeff-file /path/to/single_coeff.csv --output-file /path/to/single_output.json
"""
import pandas as pd 
import os
import glob
import json
import argparse

def analyze_collaboration(coeff_dir=None, coeff_file=None, output_file=None):
    target_names = ['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']
    measures_spec = {
        'others_influence': ['agent2'],
        'relative_behaviour': ['relative'],
        'mixed_terms': ['mixed'],
        'others_and_relative': ['agent2', 'relative'],
        'others_and_mixed': ['agent2', 'mixed'],
        'common_terms': ['mixed', 'relative'],
        'all_influences': ['agent2', 'relative', 'mixed']
    }

    def analyze_coeff_file(coeff_path, output_path=None):
        coef_df = pd.read_csv(coeff_path)
        if 'term_type' not in coef_df.columns:
            print(f"Skipping {coeff_path}: missing term_type column")
            return None
        fname = os.path.basename(coeff_path)
        parts = fname.replace('.csv','').split('_')
        training_id = parts[1] if len(parts) > 2 else 'unknown'
        checkpoint_number = parts[2] if len(parts) > 2 else 'unknown'
        total_abs = float(coef_df['abs_coeff'].sum()) if not coef_df['abs_coeff'].empty else 0.0
        n_targets = len(target_names)
        def map_types_for_target(type_list, target_idx):
            if target_idx in [0,1]:
                return type_list
            mapped = []
            for t in type_list:
                if t == 'agent2':
                    mapped.append('agent1')
                elif t == 'agent1':
                    mapped.append('agent2')
                else:
                    mapped.append(t)
            return mapped
        collaboration_measures = {}
        for m_name, m_types in measures_spec.items():
            m_score = 0.0
            m_per_target = {}
            for j_target in range(n_targets):
                tname = target_names[j_target]
                types_for_target = map_types_for_target(m_types, j_target)
                mask = coef_df['target'] == tname
                mask = mask & coef_df['term_type'].isin(types_for_target)
                s = float(coef_df.loc[mask, 'abs_coeff'].sum()) if not coef_df.loc[mask, 'abs_coeff'].empty else 0.0
                m_per_target[tname] = float(s)
                m_score += s
            m_norm = float(m_score / total_abs) if total_abs > 0 else 0.0
            collaboration_measures[m_name] = {'raw': float(m_score), 'normalized': m_norm, 'per_target_raw': m_per_target, 'per_target_normalized': {k: (v / total_abs if total_abs > 0 else 0.0) for k, v in m_per_target.items()}}
        result = {'training_id': training_id, 'checkpoint_number': checkpoint_number, 'collaboration_measures': collaboration_measures}
        if output_path:
            with open(output_path, 'w') as jf:
                json.dump(result, jf, indent=2)
            print(f"Saved collaboration measures to {output_path}")
        return result

    # Single-file mode
    if coeff_file:
        output_path = output_file or (
            f"collaboration_{os.path.basename(coeff_file).replace('.csv','.json')}")
        return analyze_coeff_file(coeff_file, output_path)

    # Batch mode (directory)
    if coeff_dir:
        coeff_files = glob.glob(os.path.join(coeff_dir, 'coefficients_*.csv'))
        results = []
        for coeff_path in coeff_files:
            fname = os.path.basename(coeff_path)
            parts = fname.replace('.csv','').split('_')
            training_id = parts[1] if len(parts) > 2 else 'unknown'
            checkpoint_number = parts[2] if len(parts) > 2 else 'unknown'
            out_path = os.path.join(coeff_dir, f'collaboration_{training_id}_{checkpoint_number}.json')
            res = analyze_coeff_file(coeff_path, out_path)
            results.append(res)
        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze collaboration from coefficients CSVs or a single file')
    parser.add_argument('--coeff-dir', type=str, help='Directory containing coefficients CSVs')
    parser.add_argument('--coeff-file', type=str, help='Single coefficients CSV to analyze')
    parser.add_argument('--output-file', type=str, help='Output JSON file for single-file mode')
    args = parser.parse_args()
    analyze_collaboration(
        coeff_dir=args.coeff_dir,
        coeff_file=args.coeff_file,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
