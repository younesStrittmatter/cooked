#!/usr/bin/env python3
"""
Test script for meaningful actions analysis.

This script tests the meaningful actions extraction on simulation data,
specifically designed to work with the updated cuttingboard verification logic.

Usage:
    nohup python testing_meaningful_actions.py --path /path/to/simulation_data > testing_meaningful_actions.log 2>&1 &
"""

import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from spoiled_broth.simulations.meaningful_actions import analyze_meaningful_actions

def test_meaningful_actions(path=None):
    """Test the meaningful actions analysis on the simulation data."""
    
    # Simulation data directory
    data_dir = Path(path) if path else print("Warning: No path provided.")
    
    # Input files
    actions_csv = data_dir / "actions.csv"
    simulation_csv = data_dir / "simulation.csv"
    
    # Output directory (same as input for this test)
    output_dir = data_dir
    
    # Map configuration
    map_name = "baseline_division_of_labor"
    engine_tick_rate = 24  # From the log file
    
    print("="*80)
    print("MEANINGFUL ACTIONS ANALYSIS TEST")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Actions CSV: {actions_csv}")
    print(f"Simulation CSV: {simulation_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Map: {map_name}")
    print(f"Engine tick rate: {engine_tick_rate} FPS")
    print()
    
    # Check if input files exist
    if not actions_csv.exists():
        print(f"ERROR: Actions CSV file not found: {actions_csv}")
        return False
    
    if not simulation_csv.exists():
        print(f"ERROR: Simulation CSV file not found: {simulation_csv}")
        return False
    
    print(f"✓ Input files exist")
    
    # Load and inspect the data
    print("\n" + "="*60)
    print("DATA INSPECTION")
    print("="*60)
    
    try:
        # Load actions data
        actions_df = pd.read_csv(actions_csv)
        print(f"Actions data shape: {actions_df.shape}")
        print(f"Actions columns: {list(actions_df.columns)}")
        print(f"Unique agents in actions: {sorted(actions_df['agent_id'].unique())}")
        print(f"Action frame range: {actions_df['action_id'].min()} to {actions_df['action_id'].max()}")
        
        # Check for NaN coordinates in actions
        nan_actions = actions_df[actions_df['target_tile_x'].isna()]
        print(f"Actions with NaN coordinates: {len(nan_actions)} ({100*len(nan_actions)/len(actions_df):.1f}%)")
        
        print()
        
        # Load simulation data  
        simulation_df = pd.read_csv(simulation_csv)
        print(f"Simulation data shape: {simulation_df.shape}")
        print(f"Simulation columns: {list(simulation_df.columns)}")
        print(f"Unique agents in simulation: {sorted(simulation_df['agent_id'].unique())}")
        print(f"Simulation frame range: {simulation_df['frame'].min()} to {simulation_df['frame'].max()}")
        
        # Check for NaN coordinates in simulation
        nan_sim = simulation_df[simulation_df['tile_x'].isna() | simulation_df['tile_y'].isna()]
        print(f"Simulation rows with NaN coordinates: {len(nan_sim)} ({100*len(nan_sim)/len(simulation_df):.1f}%)")
        
        # Check items in simulation
        items = simulation_df['item'].dropna().unique()
        print(f"Items found in simulation: {sorted(items)}")
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return False
    
    print("\n" + "="*60)
    print("RUNNING MEANINGFUL ACTIONS ANALYSIS")
    print("="*60)
    
    try:
        # Run the analysis
        result_df = analyze_meaningful_actions(
            actions_df=actions_df,
            simulation_df=simulation_df,
            map_nr=map_name,
            output_dir=output_dir,
            engine_tick_rate=engine_tick_rate
        )
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        if result_df.empty:
            print("⚠️  No meaningful actions detected!")
            return False
        
        print(f"✓ Successfully detected {len(result_df)} meaningful actions")
        
        # Analyze results by action category
        print(f"\nAction categories found:")
        category_counts = result_df['action_category_name'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        
        # Analyze by agent
        print(f"\nActions by agent:")
        agent_counts = result_df['agent_id'].value_counts()
        for agent, count in agent_counts.items():
            print(f"  {agent}: {count}")
        
        # Analyze compound actions
        compound_actions = result_df[result_df['compound_action_part'] > 0]
        if not compound_actions.empty:
            print(f"\nCompound actions: {len(compound_actions)}")
            compound_counts = compound_actions['compound_action_part'].value_counts().sort_index()
            for part, count in compound_counts.items():
                print(f"  Part {part}: {count}")
        
        # Check for cuttingboard actions
        cutting_actions = result_df[result_df['action_category_name'].str.contains('cutting|tomato_cut', na=False)]
        if not cutting_actions.empty:
            print(f"\nCuttingboard-related actions: {len(cutting_actions)}")
            for _, action in cutting_actions.iterrows():
                print(f"  Frame {action['frame']}: {action['agent_id']} - {action['action_category_name']} (part {action['compound_action_part']})")
        
        # Check for salad assembly
        assembly_actions = result_df[result_df['action_category_name'] == 'assemble salad']
        if not assembly_actions.empty:
            print(f"\nSalad assembly actions: {len(assembly_actions)}")
            for _, action in assembly_actions.iterrows():
                print(f"  Frame {action['frame']}: {action['agent_id']} - {action['previous_item']} + counter item")
        
        # Time range analysis
        print(f"\nTime analysis:")
        frame_range = result_df['frame'].max() - result_df['frame'].min()
        time_range = frame_range / engine_tick_rate
        print(f"  Frame range: {result_df['frame'].min()} to {result_df['frame'].max()} ({frame_range} frames)")
        print(f"  Time range: {time_range:.2f} seconds")
        print(f"  Actions per second: {len(result_df) / time_range:.2f}")
        
        # Check output file
        output_file = output_dir / "meaningful_actions.csv"
        if output_file.exists():
            print(f"\n✓ Output file saved: {output_file}")
            print(f"  File size: {output_file.stat().st_size} bytes")
        else:
            print(f"\n⚠️  Output file not found: {output_file}")
        
        print("\n" + "="*60)
        print("DETAILED ACTION SEQUENCE (first 20 actions)")
        print("="*60)
        
        # Show detailed sequence of first 20 actions
        for i, (_, action) in enumerate(result_df.head(20).iterrows()):
            frame = action['frame']
            second = frame / engine_tick_rate
            agent = action['agent_id']
            category = action['action_category_name']
            prev_item = action['previous_item'] or 'None'
            curr_item = action['current_item'] or 'None'
            part = action['compound_action_part']
            
            part_str = f" (part {part})" if part > 0 else ""
            print(f"{i+1:2d}. Frame {frame:4d} ({second:6.2f}s): {agent} - {prev_item} → {curr_item} | {category}{part_str}")
        
        if len(result_df) > 20:
            print(f"    ... and {len(result_df) - 20} more actions")
        
        return True
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    # Obtain path from command line argument
    parser = argparse.ArgumentParser(description="Test meaningful actions analysis on simulation data.")
    parser.add_argument('--path', type=str, required=True,
                        help="Path to the simulation data directory")
    args = parser.parse_args()

    print("Starting meaningful actions test...")

    success = test_meaningful_actions(args.path)

    if success:
        print("\n" + "="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print("The meaningful actions analysis worked correctly!")
        print("Check the output file for detailed results.")
    else:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()