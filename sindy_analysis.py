#!/usr/bin/env python3
"""
Detailed SINDy analysis over experiment data in /data/samuel_lozano/cooked/classic/v3.1/experiment/

- Runs SINDy for each (training_id, checkpoint_number)
- Saves equations and analysis to /experiments/map_{map_identifier}/
- Analyzes collaboration/influence via cross-agent terms
- Validates equations (accuracy, RMSE, R²)
- Plots average prediction vs. true and error distributions

Usage:
    nohup python3 sindy_analysis.py <map_identifier> [--base-path /path/to/experiment] [--output-path /path/to/output] > log_sindy_analysis.txt 2>&1 &

    nohup python3 sindy_analysis.py baseline_division_of_labor > log_sindy_analysis.out 2>&1 &

Requires:
    pip install pysindy numpy pandas matplotlib
"""

import argparse
from pathlib import Path
from spoiled_broth.sindy.extract_equations import extract_equations
from spoiled_broth.sindy.analyze_collaboration import analyze_collaboration
from spoiled_broth.sindy.plot_and_summarize import plot_and_summarize

def main():
    parser = argparse.ArgumentParser(description='Orchestrate SINDy analysis pipeline')
    parser.add_argument('map_identifier', type=str, help='Map identifier (e.g. baseline_division_of_labor)')
    parser.add_argument('--base-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment', help='Base path to experiment folder')
    parser.add_argument('--threshold', type=float, default=1e-4, help='STLSQ threshold')
    parser.add_argument('--poly-degree', type=int, default=2, help='Polynomial library degree')
    parser.add_argument('--fourier-n', type=int, default=0, help='Number of Fourier components to add (0 = none)')
    args = parser.parse_args()

    # Step 1: Extract equations and coefficients
    print("Step 1: Extracting equations and coefficients...")
    extract_equations(
        map_identifier=args.map_identifier,
        base_path=args.base_path,
        threshold=args.threshold,
        poly_degree=args.poly_degree,
        fourier_n=args.fourier_n
    )

    # Step 2: Analyze collaboration
    print("Step 2: Analyzing collaboration...")
    coeff_dir = str(Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / f"threshold_{args.threshold}-poly_{args.poly_degree}-fourier_{args.fourier_n}")
    collab_out_dir = coeff_dir
    analyze_collaboration(
        coeff_dir=coeff_dir,
        output_dir=collab_out_dir
    )

    # Step 3: Plot and summarize
    print("Step 3: Plotting and summarizing...")
    summary_path = str(Path(coeff_dir) / f'summary_{args.map_identifier}.json')
    plot_and_summarize(
        collab_dir=collab_out_dir,
        output_path=summary_path
    )
    print(f"Pipeline complete. See {summary_path} for results.")

if __name__ == "__main__":
    main()
