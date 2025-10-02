#!/usr/bin/env python3
"""
Improved action and logger comparison with better diagnostics.

This script addresses the core issues found in the action tracker:
1. Compares actual completion positions instead of target positions
2. Accounts for coordinate system consistency
3. Provides detailed analysis of position tracking accuracy
"""

import pandas as pd
import numpy as np
import sys
import os

def load_data(actions_csv, frames_csv):
    """Load action and frame data with error handling."""
    try:
        actions = pd.read_csv(actions_csv)
        frames = pd.read_csv(frames_csv)
        print(f"Loaded {len(actions)} actions and {len(frames)} frame records")
        return actions, frames
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_coordinate_consistency(actions, frames):
    """Analyze coordinate system consistency between action and frame logs."""
    print("\n=== COORDINATE SYSTEM ANALYSIS ===")
    
    # Check tile coordinate calculation consistency
    for _, action in actions.iterrows():
        # Find corresponding frame data
        agent_id = action['agent_id']
        completion_frame = action['completion_frame']
        
        frame_data = frames[
            (frames['agent_id'] == agent_id) & 
            (frames['frame'] == completion_frame)
        ]
        
        if not frame_data.empty:
            frame_row = frame_data.iloc[0]
            
            # Calculate tile coordinates using both methods
            method1_x = int(frame_row['x'] // 16 + 1)  # Current frame method
            method1_y = int(frame_row['y'] // 16 + 1)
            
            method2_x = float(frame_row['x']) // 16 + 1  # Old frame method  
            method2_y = float(frame_row['y']) // 16 + 1
            
            logged_tile_x = frame_row['tile_x']
            logged_tile_y = frame_row['tile_y']
            
            action_tile_x = action['completion_tile_x']
            action_tile_y = action['completion_tile_y']
            
            # Check for inconsistencies
            if (method1_x != logged_tile_x or method1_y != logged_tile_y or
                method1_x != action_tile_x or method1_y != action_tile_y):
                print(f"Coordinate inconsistency found:")
                print(f"  Agent: {agent_id}, Frame: {completion_frame}")
                print(f"  Position: ({frame_row['x']}, {frame_row['y']})")
                print(f"  Method 1 (int): ({method1_x}, {method1_y})")
                print(f"  Method 2 (float): ({method2_x}, {method2_y})")
                print(f"  Frame log: ({logged_tile_x}, {logged_tile_y})")
                print(f"  Action log: ({action_tile_x}, {action_tile_y})")
                print()
                
def check_position_accuracy(actions, frames):
    """Check accuracy of position tracking between systems."""
    discrepancies = []
    position_errors = []
    tile_errors = []
    
    print("\n=== POSITION ACCURACY ANALYSIS ===")
    
    for idx, action in actions.iterrows():
        agent_id = action['agent_id']
        start_frame = action['decision_frame']
        end_frame = action['completion_frame']
        
        # Get frame data for this action period
        frame_subset = frames[
            (frames['agent_id'] == agent_id) &
            (frames['frame'] >= start_frame) &
            (frames['frame'] <= end_frame)
        ]
        
        if frame_subset.empty:
            discrepancies.append({
                "agent_id": agent_id,
                "frame_range": f"({start_frame}, {end_frame})",
                "issue": "No tracking data for this frame range",
                "frame": agent_id
            })
            continue
        
        # Get final frame data
        final_frame = frame_subset.iloc[-1]
        
        # Position comparison
        tracked_pos = np.array([final_frame['x'], final_frame['y']])
        logged_pos = np.array([action['completion_x'], action['completion_y']])
        position_distance = np.linalg.norm(tracked_pos - logged_pos)
        
        if position_distance > 5.0:  # 5 pixel tolerance
            position_errors.append(position_distance)
            discrepancies.append({
                "agent_id": agent_id,
                "frame_range": f"({start_frame}, {end_frame})",
                "issue": f"Completion pos mismatch: log={logged_pos}, tracked={tracked_pos}, dist={position_distance:.2f}",
                "frame": ""
            })
        
        # Tile comparison (use completion tiles, not target tiles)
        tracked_tile = (final_frame['tile_x'], final_frame['tile_y'])
        logged_completion_tile = (action['completion_tile_x'], action['completion_tile_y'])
        
        if tracked_tile != logged_completion_tile:
            tile_errors.append(1)
            discrepancies.append({
                "agent_id": agent_id,
                "frame_range": f"({start_frame}, {end_frame})",
                "issue": f"Tile mismatch: log={logged_completion_tile}, tracked={tracked_tile}",
                "frame": ""
            })
        
        # Target vs completion analysis (this is expected to have differences)
        target_tile = (action.get('target_tile_x', 0), action.get('target_tile_y', 0))
        if target_tile != logged_completion_tile:
            # This is normal - agents don't always reach exact targets
            pass
    
    # Print statistics
    print(f"Total actions analyzed: {len(actions)}")
    print(f"Position errors (>5px): {len(position_errors)}")
    print(f"Tile mismatches: {len(tile_errors)}")
    
    if position_errors:
        print(f"Average position error: {np.mean(position_errors):.2f}px")
        print(f"Max position error: {np.max(position_errors):.2f}px")
    
    return pd.DataFrame(discrepancies)

def analyze_action_patterns(actions, frames):
    """Analyze patterns in action execution and completion."""
    print("\n=== ACTION PATTERN ANALYSIS ===")
    
    # Analyze action durations
    durations = actions['duration_frames']
    print(f"Action duration stats:")
    print(f"  Mean: {durations.mean():.1f} frames")
    print(f"  Median: {durations.median():.1f} frames") 
    print(f"  Min: {durations.min()} frames")
    print(f"  Max: {durations.max()} frames")
    
    # Analyze movement distances
    actions['movement_distance'] = np.sqrt(
        (actions['completion_x'] - actions['decision_x'])**2 +
        (actions['completion_y'] - actions['decision_y'])**2
    )
    
    movements = actions['movement_distance'].dropna()
    print(f"\nMovement distance stats:")
    print(f"  Mean: {movements.mean():.1f}px")
    print(f"  Median: {movements.median():.1f}px")
    print(f"  Max: {movements.max():.1f}px")
    
    # Find actions with large movements
    large_movements = actions[actions['movement_distance'] > 32]  # More than 2 tiles
    if not large_movements.empty:
        print(f"\nActions with large movements (>{32}px):")
        for _, action in large_movements.iterrows():
            print(f"  {action['agent_id']}: {action['movement_distance']:.1f}px in {action['duration_frames']} frames")

def main():
    """Main analysis function."""
    # Configuration
    simulation_id = "2025_10_02-03_04_19"
    training_map_nr = "baseline_division_of_labor"
    game_version = "classic"
    intent_version = "v3.1"
    training_id = "2025-09-19_14-40-19"
    checkpoint_number = 62500
    
    # File paths
    base_dir = f"/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{training_map_nr}/cooperative/Training_{training_id}/simulations/simulations_{checkpoint_number}/simulation_{simulation_id}"
    actions_csv = f"{base_dir}/actions.csv"
    frames_csv = f"{base_dir}/simulation.csv"
    output_csv = f"{base_dir}/improved_discrepancies.csv"
    
    # Check if files exist
    if not os.path.exists(actions_csv) or not os.path.exists(frames_csv):
        print("Error: Could not find required CSV files")
        print(f"Looking for:")
        print(f"  Actions: {actions_csv}")
        print(f"  Frames: {frames_csv}")
        sys.exit(1)
    
    # Load data
    actions, frames = load_data(actions_csv, frames_csv)
    
    # Run analyses
    analyze_coordinate_consistency(actions, frames)
    discrepancies = check_position_accuracy(actions, frames)
    analyze_action_patterns(actions, frames)
    
    # Save results
    if not discrepancies.empty:
        discrepancies.to_csv(output_csv, index=False)
        print(f"\nDetailed discrepancies saved to: {output_csv}")
        print(f"Total discrepancies found: {len(discrepancies)}")
    else:
        print("\nNo discrepancies found!")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. The action tracker should record completion positions more accurately")
    print("2. Consider increasing position tolerance for moving agents")
    print("3. Tile coordinate calculation should be consistent across all systems")
    print("4. Target vs completion position differences are normal and expected")

if __name__ == "__main__":
    main()