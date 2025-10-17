#!/usr/bin/env python3
"""
Step 2: Analyze collaboration from coefficients CSVs.
- Loads coefficients CSV, computes 7 collaboration measures per checkpoint.
- Saves results as collaboration_<training>_<checkpoint>.json

Usage:
    nohup python3 analyze_collaboration.py --coeff-dir /path/to/coeffs > log_analyze_collaboration.out 2>&1 &
    
    nohup python3 analyze_collaboration.py --coeff-file /path/to/single_coeff.csv --output-file /path/to/single_output.json
"""
from pathlib import Path
import pandas as pd 
import os
import glob
import json
import argparse

 # Infer term_type for each row
def infer_term_type(row):
    term = row['term']
    term_parts = term.split(' ') if ' ' in term else [term]
    term_agents = set()
    has_rel = False
    for part in term_parts:
        part = part.strip()
        if part.startswith('1'):
            term_agents.add('own')
        if 'rel_' in part:
            has_rel = True
        if '_1' in part:
            term_agents.add('agent1')
        elif '_2' in part:
            term_agents.add('agent2')
    if len(term_parts) == 1:
        if term_parts[0].endswith('_1'):
            return 'agent1'
        elif term_parts[0].endswith('_2'):
            return 'agent2'
    if 'agent1' in term_agents and 'agent2' in term_agents:
        return 'mixed'
    elif has_rel:
        return 'relative'
    elif 'agent1' in term_agents:
        return 'agent1'
    elif 'agent2' in term_agents:
        return 'agent2'
    elif 'own' in term_agents:
        return 'own'
    return 'unknown'

# Infer target_type for each row
def infer_target_type(target):
    if str(target).endswith('_1'):
        return 'agent1'
    elif str(target).endswith('_2'):
        return 'agent2'
    return 'unknown'

# Define measure masks based on term_type and target_type
def measure_mask(df, measure):
    if measure == 'own_influence':
        mask1 = (df['term_type'] == 'own') & ((df['target_type'] == 'agent1') | (df['target_type'] == 'agent2'))
        mask2 = (df['term_type'] == 'agent1') & (df['target_type'] == 'agent1')
        mask3 = (df['term_type'] == 'agent2') & (df['target_type'] == 'agent2')
        return mask1 | mask2 | mask3
    elif measure == 'others_influence':
        return ((df['term_type'] == 'agent1') & (df['target_type'] == 'agent2')) | ((df['term_type'] == 'agent2') & (df['target_type'] == 'agent1'))
    elif measure == 'relative_behaviour':
        return df['term_type'] == 'relative'
    elif measure == 'mixed_terms':
        return df['term_type'] == 'mixed'
    else:
        return pd.Series([False]*len(df))

# Analyze a single coefficients CSV file
def analyze_coeff_file(map_identifier, coeff_path, target_names, output_path=None):
    coef_df = pd.read_csv(coeff_path)
    fname = os.path.basename(coeff_path)
    parts = fname.replace('.csv','').split('_')
    if len(parts) >= 4:
        training_id = f"{parts[1]}_{parts[2]}"
        checkpoint_number = parts[3]
    else:
        training_id = 'unknown'
        checkpoint_number = 'unknown'
    total_abs = float(coef_df['abs_coeff'].sum()) if not coef_df['abs_coeff'].empty else 0.0

    # Add inferred term_type and target_type columns
    coef_df['term_type'] = coef_df.apply(infer_term_type, axis=1)
    coef_df['target_type'] = coef_df['target'].apply(infer_target_type)

    # Calculate base metrics first
    base_metrics = ['own_influence', 'others_influence', 'relative_behaviour', 'mixed_terms']
    base_scores = {}
    base_per_target = {}
    for m_name in base_metrics:
        m_score = 0.0
        m_per_target = {}
        for tname in target_names:
            mask = (coef_df['target'] == tname) & measure_mask(coef_df, m_name)
            s = float(coef_df.loc[mask, 'abs_coeff'].sum()) if not coef_df.loc[mask, 'abs_coeff'].empty else 0.0
            m_per_target[tname] = float(s)
            m_score += s
        base_scores[m_name] = m_score
        base_per_target[m_name] = m_per_target

    # Normalize so that the four base metrics add up to 1
    total_base = sum(base_scores.values())
    base_norm = {k: (v / total_base if total_base > 0 else 0.0) for k, v in base_scores.items()}
    base_per_target_norm = {k: {t: (v / total_base if total_base > 0 else 0.0) for t, v in base_per_target[k].items()} for k in base_metrics}

    # Composite metrics as sums of normalized base metrics
    composite_metrics = {
        'others_and_relative': ['others_influence', 'relative_behaviour'],
        'others_and_mixed': ['others_influence', 'mixed_terms'],
        'common_terms': ['relative_behaviour', 'mixed_terms'],
        'all_influences': ['others_influence', 'relative_behaviour', 'mixed_terms']
    }

    collaboration_measures = {}
    # Add base metrics
    for m_name in base_metrics:
        collaboration_measures[m_name] = {
            'raw': float(base_scores[m_name]),
            'normalized': base_norm[m_name],
            'per_target_raw': base_per_target[m_name],
            'per_target_normalized': base_per_target_norm[m_name]
        }
    # Add composite metrics
    for m_name, components in composite_metrics.items():
        raw_sum = sum(base_scores[c] for c in components)
        norm_sum = sum(base_norm[c] for c in components)
        per_target_raw = {t: sum(base_per_target[c][t] for c in components) for t in target_names}
        per_target_norm = {t: sum(base_per_target_norm[c][t] for c in components) for t in target_names}
        collaboration_measures[m_name] = {
            'raw': float(raw_sum),
            'normalized': norm_sum,
            'per_target_raw': per_target_raw,
            'per_target_normalized': per_target_norm
        }

    result = {'map': map_identifier, 'training_id': training_id, 'checkpoint_number': checkpoint_number, 'collaboration_measures': collaboration_measures}
    if output_path:
        with open(output_path, 'w') as jf:
            json.dump(result, jf, indent=2)
        print(f"Saved collaboration measures to {output_path}")
    return result

# Main function to handle command-line arguments
def analyze_collaboration(map_identifier, coeff_dir=None, coeff_file=None, output_file=None):
    target_names = ['accel_x_1', 'accel_y_1', 'accel_x_2', 'accel_y_2']

    # Single-file mode
    if coeff_file:
        output_path = output_file or (
            f"collaboration_{os.path.basename(coeff_file).replace('.csv','.json')}")
        return analyze_coeff_file(coeff_file, target_names, output_path)

    # Batch mode (directory)
    if coeff_dir:
        coeff_files = glob.glob(os.path.join(coeff_dir, 'coefficients_*.csv'))
        results = []
        for coeff_path in coeff_files:
            fname = os.path.basename(coeff_path)
            parts = fname.replace('.csv','').split('_')
            if len(parts) >= 4:
                training_id = f"{parts[1]}_{parts[2]}"
                checkpoint_number = parts[3]
            else:
                training_id = 'unknown'
                checkpoint_number = 'unknown'
            out_path = os.path.join(coeff_dir, f'collaboration_{training_id}_{checkpoint_number}.json')
            res = analyze_coeff_file(map_identifier, coeff_path, target_names, out_path)
            results.append(res)
        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze collaboration from coefficients CSVs or a single file')
    parser.add_argument('map_identifier', type=str)
    parser.add_argument('--base-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment')
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--poly-degree', type=int, default=2)
    parser.add_argument('--fourier-n', type=int, default=0)
    parser.add_argument('--coeff-file', type=str, help='Single coefficients CSV file to analyze')
    parser.add_argument('--output-file', type=str, help='Output JSON file for single coefficients file analysis')
    args = parser.parse_args()

    # compute directories
    coeff_dir = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / f"threshold_{args.threshold}-poly_{args.poly_degree}-fourier_{args.fourier_n}"
    coeff_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / 'standardized_data'
    data_path.mkdir(parents=True, exist_ok=True)

    analyze_collaboration(
        map_identifier=args.map_identifier,
        coeff_dir=coeff_dir,
        coeff_file=args.coeff_file,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
