"""
Data extraction utilities for human-like analysis of simulation data.

This module provides functions to extract and format simulation data
in formats commonly used by human researchers for analysis.

Author: Samuel Lozano
"""

import pandas as pd
import math
from pathlib import Path


def generate_agent_position_files(simulation_df, output_dir, agent_initialization_period=0.0):
    """
    Generate position CSV files for each agent from simulation data.
    
    Creates individual CSV files for each agent with columns:
    second, x, y, tile_x, tile_y, distance_walked, walking_speed, cutting_speed, 
    start_pos, item, score, frame
    
    The function filters out the initialization period and adjusts time so that
    the first second after initialization becomes second 0.
    
    Args:
        simulation_df: DataFrame with simulation data or path to CSV file
        output_dir: Directory to save the position files
        agent_initialization_period: Duration of agent initialization period in seconds (default 15.0)
        
    Returns:
        Dictionary with paths to generated position files {agent_id: filepath}
    """
    
    # Read the CSV file if it's a path, otherwise use the dataframe directly
    if isinstance(simulation_df, (str, Path)):
        simulation_df = pd.read_csv(simulation_df)
    
    output_dir = Path(output_dir)
    
    # Try to read actual agents start acting frame/time from config file if available
    actual_start_time = None
    config_file = output_dir / "config.txt"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    if line.strip().startswith("ACTUAL_AGENTS_START_ACTING_TIME:"):
                        actual_start_time = float(line.split(":", 1)[1].strip())
                        
            if actual_start_time is not None:
                # Use the minimum between runtime-detected time and initialization period
                filter_time = min(actual_start_time, agent_initialization_period)
                print(f"Runtime-detected agents start time: {actual_start_time:.3f}s")
                print(f"Agent initialization period: {agent_initialization_period:.3f}s")
                print(f"Using minimum filter time: {filter_time:.3f}s")
            else:
                print(f"Runtime detection not found in config, using agent_initialization_period: {agent_initialization_period}s")
                filter_time = agent_initialization_period
        except Exception as e:
            print(f"Warning: Could not read config file {config_file}: {e}")
            print(f"Using fallback agent_initialization_period: {agent_initialization_period}s")
            filter_time = agent_initialization_period
    else:
        print(f"Config file not found, using agent_initialization_period: {agent_initialization_period}s")
        filter_time = agent_initialization_period
    
    position_files = {}
    
    # Process each agent
    for agent_id in simulation_df['agent_id'].unique():
        if not agent_id.startswith('ai_rl_'):
            continue
            
        agent_data = simulation_df[simulation_df['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('frame').reset_index(drop=True)
        
        # Filter out initialization period (keep only data after agents actually start acting)
        agent_data = agent_data[agent_data['second'] >= filter_time].copy()
        
        if agent_data.empty:
            print(f"Warning: No data for {agent_id} after filter time of {filter_time:.3f} seconds")
            continue
        
        # Adjust time so that first second after filter becomes second 0
        agent_data['second'] = agent_data['second'] - filter_time
        agent_data = agent_data.reset_index(drop=True)
        
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
        
        # Get start position (tile coordinates from first frame after initialization)
        if not agent_data.empty:
            start_tile_x = agent_data.iloc[0]['tile_x']
            start_tile_y = agent_data.iloc[0]['tile_y']
            start_pos = f"({start_tile_x}, {start_tile_y})"
        else:
            start_pos = "(0, 0)"  # Fallback if no data available
        
        # Create the position dataframe with required columns
        position_data = pd.DataFrame({
            'second': agent_data['second'],
            'x': agent_data['x'],
            'y': agent_data['y'],
            'tile_x': agent_data['tile_x'],
            'tile_y': agent_data['tile_y'],
            'distance_walked': agent_data['distance_walked'],
            'walking_speed': 1.0,
            'cutting_speed': 1.0,
            'start_pos': start_pos,
            'item': agent_data['item'],
            'score': agent_data['score'],
            'frame': agent_data['frame']
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
        print(f"  Final score: {position_data['score'].iloc[-1]}")
        print(f"  Items held: {position_data['item'].nunique()} unique items")
    
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