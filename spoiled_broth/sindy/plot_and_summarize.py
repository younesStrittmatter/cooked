#!/usr/bin/env python3
"""
Step 3: Plot figures and create summary JSON.
- Loads results, generates plots, writes summary_<training>_<checkpoint>.json and FINAL_RESULTS.json

Usage:
    nohup python3 plot_and_summarize.py --collab-dir /path/to/collaboration_jsons --output-path /path/to/summary.json > log_plot_and_summarize.out 2>&1 &
"""
import os, glob, json
import numpy as np
import argparse

def plot_and_summarize(collab_dir, output_path):
    collab_files = glob.glob(os.path.join(collab_dir, 'collaboration_*.json'))
    all_results = []
    for f in collab_files:
        with open(f, 'r') as jf:
            res = json.load(jf)
            all_results.append(res)
    # Aggregate measures
    measures = list(all_results[0]['collaboration_measures'].keys()) if all_results else []
    summary = {'n_checkpoints': len(all_results), 'per_checkpoint': all_results}
    for m in measures:
        vals = [r['collaboration_measures'][m]['normalized'] for r in all_results if m in r['collaboration_measures']]
        summary[f'avg_{m}_normalized'] = float(np.mean(vals)) if vals else None
    # Optionally: plot measures
    # ... (add plotting code if needed) ...
    with open(output_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved summary to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot and summarize collaboration results')
    parser.add_argument('--collab-dir', type=str, required=True, help='Directory containing collaboration JSONs')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save summary JSON')
    args = parser.parse_args()
    plot_and_summarize(
        collab_dir=args.collab_dir,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
