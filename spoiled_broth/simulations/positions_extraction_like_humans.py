"""
Data extraction utilities for human-like analysis of simulation data.

This module provides functions to extract and format simulation data
in formats commonly used by human researchers for analysis.

Author: Samuel Lozano
"""

import pandas as pd
import math
from pathlib import Path


def generate_agent_position_files(simulation_df, output_dir):
    """
    Generate position CSV files for each agent from simulation data.
    
    Creates individual CSV files for each agent with columns:
    second, x, y, distance_walked, walking_speed, cutting_speed, start_pos
    
    Args:
        simulation_df: DataFrame with simulation data or path to CSV file
        output_dir: Directory to save the position files
        
    Returns:
        Dictionary with paths to generated position files {agent_id: filepath}
    """
    
    # Read the CSV file if it's a path, otherwise use the dataframe directly
    if isinstance(simulation_df, (str, Path)):
        simulation_df = pd.read_csv(simulation_df)
    
    output_dir = Path(output_dir)
    position_files = {}
    
    # Process each agent
    for agent_id in simulation_df['agent_id'].unique():
        if not agent_id.startswith('ai_rl_'):
            continue
            
        agent_data = simulation_df[simulation_df['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('frame').reset_index(drop=True)
        
        # Calculate distance walked
        agent_data['distance_walked'] = 0.0
        
        if len(agent_data) > 1:
            # Calculate cumulative distance
            for i in range(1, len(agent_data)):
                prev_x = agent_data.loc[i-1, 'x']
                prev_y = agent_data.loc[i-1, 'y']
                curr_x = agent_data.loc[i, 'x']
                curr_y = agent_data.loc[i, 'y']
                
                # Calculate Euclidean distance
                distance_step = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                agent_data.loc[i, 'distance_walked'] = agent_data.loc[i-1, 'distance_walked'] + distance_step
        
        # Get start position (tile coordinates at frame 0)
        start_frame_data = agent_data[agent_data['frame'] == 0]
        if not start_frame_data.empty:
            start_tile_x = start_frame_data.iloc[0]['tile_x']
            start_tile_y = start_frame_data.iloc[0]['tile_y']
            start_pos = f"({start_tile_x}, {start_tile_y})"
        else:
            # Fallback to first available frame
            start_tile_x = agent_data.iloc[0]['tile_x']
            start_tile_y = agent_data.iloc[0]['tile_y']
            start_pos = f"({start_tile_x}, {start_tile_y})"
        
        # Create the position dataframe with required columns
        position_data = pd.DataFrame({
            'second': agent_data['second'],
            'x': agent_data['x'],
            'y': agent_data['y'],
            'distance_walked': agent_data['distance_walked'],
            'walking_speed': 1.0,
            'cutting_speed': 1.0,
            'start_pos': start_pos
        })
        
        # Save to CSV
        filename = f"{agent_id}_positions.csv"
        filepath = output_dir / filename
        position_data.to_csv(filepath, index=False)
        position_files[agent_id] = filepath
        
        print(f"Generated position file for {agent_id}: {filepath}")
        print(f"  Total frames: {len(position_data)}")
        print(f"  Total distance: {position_data['distance_walked'].iloc[-1]:.2f} pixels")
        print(f"  Start position: {start_pos}")
    
    return position_files


def extract_agent_trajectories(simulation_df, output_dir=None, save_individual=True):
    """
    Extract agent movement trajectories with detailed position and timing data.
    
    Args:
        simulation_df: DataFrame with simulation data or path to CSV file
        output_dir: Directory to save files (optional)
        save_individual: Whether to save individual agent files
        
    Returns:
        Dictionary with trajectory data for each agent
    """
    
    # Read the CSV file if it's a path, otherwise use the dataframe directly
    if isinstance(simulation_df, (str, Path)):
        simulation_df = pd.read_csv(simulation_df)
    
    trajectories = {}
    
    for agent_id in simulation_df['agent_id'].unique():
        if not agent_id.startswith('ai_rl_'):
            continue
            
        agent_data = simulation_df[simulation_df['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('frame').reset_index(drop=True)
        
        # Calculate velocity and acceleration
        agent_data['velocity_x'] = 0.0
        agent_data['velocity_y'] = 0.0
        agent_data['speed'] = 0.0
        agent_data['acceleration'] = 0.0
        
        if len(agent_data) > 1:
            for i in range(1, len(agent_data)):
                dt = agent_data.loc[i, 'second'] - agent_data.loc[i-1, 'second']
                if dt > 0:
                    dx = agent_data.loc[i, 'x'] - agent_data.loc[i-1, 'x']
                    dy = agent_data.loc[i, 'y'] - agent_data.loc[i-1, 'y']
                    
                    agent_data.loc[i, 'velocity_x'] = dx / dt
                    agent_data.loc[i, 'velocity_y'] = dy / dt
                    agent_data.loc[i, 'speed'] = math.sqrt(dx*dx + dy*dy) / dt
                    
                    if i > 1:
                        prev_speed = agent_data.loc[i-1, 'speed']
                        agent_data.loc[i, 'acceleration'] = (agent_data.loc[i, 'speed'] - prev_speed) / dt
        
        trajectories[agent_id] = agent_data
        
        # Save individual trajectory file if requested
        if save_individual and output_dir:
            output_dir = Path(output_dir)
            filename = f"{agent_id}_trajectory.csv"
            filepath = output_dir / filename
            agent_data.to_csv(filepath, index=False)
            print(f"Generated trajectory file for {agent_id}: {filepath}")
    
    return trajectories