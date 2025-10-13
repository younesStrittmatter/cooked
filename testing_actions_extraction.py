#!/usr/bin/env python3
"""
Script to run actions extraction over simulation data with meaningful actions.

Usage:
    nohup python testing_actions_extraction.py --path /path/to/simulation_data > testing_actions_extraction.log 2>&1 &
"""

import sys
import pandas as pd
from pathlib import Path
import argparse

# Add the current directory to Python path so we can import our modules
sys.path.append('/home/samuel_lozano/cooked')

from spoiled_broth.simulations.actions_extraction_like_humans import generate_agent_action_files

def run_actions_extraction(path=None):
    """Run the actions extraction over the simulation data."""
    
    # Define paths
    data_dir = Path(path) if path else print("Warning: No path provided.")
    
    # Input files
    meaningful_actions_csv = data_dir / "meaningful_actions.csv"
    positions_dir = data_dir  # Assuming position files are in the same directory
    
    # Output directory
    output_dir = data_dir
    
    # Check if meaningful_actions.csv exists
    if not meaningful_actions_csv.exists():
        print(f"Error: meaningful_actions.csv not found at {meaningful_actions_csv}")
        print("Please run meaningful_actions.py first to generate this file.")
        return
    
    print(f"Using meaningful actions CSV: {meaningful_actions_csv}")
    print(f"Looking for position files in: {positions_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check for position files
    position_files = list(positions_dir.glob("*_positions.csv"))
    if not position_files:
        print(f"Warning: No position files found in {positions_dir}")
        print("Looking for files matching pattern: *_positions.csv")
        # List all CSV files to help debug
        all_csv_files = list(positions_dir.glob("*.csv"))
        print(f"Available CSV files: {[f.name for f in all_csv_files]}")
    else:
        print(f"Found position files: {[f.name for f in position_files]}")
    
    try:
        # Load meaningful actions
        print("\nLoading meaningful actions...")
        meaningful_actions_df = pd.read_csv(meaningful_actions_csv)
        print(f"Loaded {len(meaningful_actions_df)} meaningful actions")
        print(f"Agents found: {meaningful_actions_df['agent_id'].unique()}")
        
        # Run the actions extraction
        print("\nGenerating agent action files...")
        action_files = generate_agent_action_files(
            meaningful_actions_df=meaningful_actions_df,
            positions_dir=positions_dir,
            output_dir=output_dir,
            map_name="baseline_division_of_labor",
            simulation_id="simulation_2025_10_10-04_56_00",
            engine_tick_rate=24,  # Default engine tick rate
            agent_initialization_period=0.0
        )
        
        print(f"\nGenerated action files:")
        for agent_id, filepath in action_files.items():
            print(f"  {agent_id}: {filepath}")
            
        if action_files:
            print(f"\nSuccess! Generated {len(action_files)} action files in {output_dir}")
        else:
            print("\nWarning: No action files were generated. Check the input data and file paths.")
            
    except Exception as e:
        print(f"Error running actions extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run actions extraction on simulation data.")
    parser.add_argument('--path', type=str, required=True,
                        help="Path to the simulation data directory")
    args = parser.parse_args()

    print("Starting actions extraction test...")

    run_actions_extraction(path=args.path)

    print("\n" + "="*80)
    print("✅ TEST COMPLETED")
    print("="*80)
    print("The actions extraction process has completed.")
    print("Check the output files for detailed results.")