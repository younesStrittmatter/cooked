#!/usr/bin/env python3
"""
Post-process SINDy threshold optimization results.
Given a folder containing results.json, computes (r2+1)/2 + 1/(1+mse) for each config and selects the best.
Saves best.json and prints the best config.

Usage:
    python3 postprocess_sindy_results.py --results-dir /path/to/threshold_optimization
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def postprocess_results(results_dir, weight_norm_r2=0.5, weight_norm_mse=0.5):
    results_path = Path(results_dir) / 'results.json'
    if not results_path.exists():
        print(f"results.json not found in {results_dir}")
        return
    with open(results_path, 'r') as f:
        loaded_results = json.load(f)
    best = None
    best_score = -float('inf')
    best_val = None
    plot_data = []
    for r in loaded_results:
        if r.get('error') is not None:
            continue
        mean_r2 = r.get('mean_r2')
        mean_mse = r.get('mean_mse')
        if mean_r2 is None or mean_mse is None:
            continue
        norm_r2 = (mean_r2 + 1.0) / 2.0
        norm_mse = 1.0 / (1.0 + mean_mse)
        score = weight_norm_r2 * norm_r2 + weight_norm_mse * norm_mse
        r['normalized_r2'] = norm_r2
        r['normalized_mse'] = norm_mse
        r['combined_score'] = score
        plot_data.append({
            'label': f"thr={r['threshold']},poly={r['poly_degree']},fourier={r['fourier_n']}",
            'combined_score': score,
            'normalized_r2': norm_r2,
            'normalized_mse': norm_mse
        })
        if score > best_score:
            best_score = score
            best = r
            best_val = score
    if best:
        print(f"Best config: threshold={best['threshold']}, poly_degree={best['poly_degree']}, fourier_n={best['fourier_n']}, combined_score={best_val}")
        with open(Path(results_dir) / 'best.json', 'w') as f:
            json.dump(best, f, indent=2)
    else:
        print("No valid scores found for any configuration.")

    # Plot comparison scatter plots (sorted)
    try:
        import matplotlib.pyplot as plt
        # Sort by combined_score
        plot_data_sorted = sorted(plot_data, key=lambda d: d['combined_score'])
        labels = [d['label'] for d in plot_data_sorted]
        x = range(len(labels))
        fontsize = 8
        # Combined score scatter
        plt.figure(figsize=(max(8, len(labels)*0.5), 4))
        y = [d['combined_score'] for d in plot_data_sorted]
        plt.plot(x, y, marker='o', color='dodgerblue', linestyle='-', linewidth=1)
        plt.xticks(x, labels, rotation=90, fontsize=fontsize)
        plt.ylabel('Combined Score (normalized_r2 + normalized_mse)')
        plt.title('SINDy Config Comparison: Combined Score')
        plt.ylim(min(y)-0.05, max(y)+0.05)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / 'combined_score_scatter.png')
        plt.close()

        # Normalized R2 scatter
        plt.figure(figsize=(max(8, len(labels)*0.5), 4))
        y_r2 = [d['normalized_r2'] for d in plot_data_sorted]
        plt.plot(x, y_r2, marker='o', color='orange', linestyle='-', linewidth=1)
        plt.xticks(x, labels, rotation=90, fontsize=fontsize)
        plt.ylabel('Normalized R2')
        plt.title('SINDy Config Comparison: Normalized R2')
        plt.ylim(min(y_r2)-0.05, max(y_r2)+0.05)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / 'normalized_r2_scatter.png')
        plt.close()

        # Normalized MSE scatter
        plt.figure(figsize=(max(8, len(labels)*0.5), 4))
        y_mse = [d['normalized_mse'] for d in plot_data_sorted]
        plt.plot(x, y_mse, marker='o', color='green', linestyle='-', linewidth=1)
        plt.xticks(x, labels, rotation=90, fontsize=fontsize)
        plt.ylabel('Normalized MSE')
        plt.title('SINDy Config Comparison: Normalized MSE')
        plt.ylim(min(y_mse)-0.05, max(y_mse)+0.05)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / 'normalized_mse_scatter.png')
        plt.close()
        print("Saved comparison scatter plots in", results_dir)
    except Exception as e:
        print("Could not generate plots:", e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process SINDy results.json to select best config')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing results.json')
    args = parser.parse_args()
    postprocess_results(args.results_dir)
