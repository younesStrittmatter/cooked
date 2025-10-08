# USE: nohup python spoiled_broth/simulations/meaningful_actions.py --actions_csv spoiled_broth/simulations/actions.csv --simulation_csv spoiled_broth/simulations/simulation_log.csv --map 1 --output_dir spoiled_broth/simulations/ > log_meaningful_actions.out 2>&1 &

"""
Meaningful actions analysis for simulation runs.

This module provides functionality to detect meaningful actions by finding 
item state changes and matching them to actions.csv data.

Author: Samuel Lozano
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path

def load_map_info(map_nr):
    """Load map information to understand what each dispenser provides"""
    # Try to read the map file to understand dispenser types
    map_file = Path(__file__).parent.parent / 'maps' / f'{map_nr}.txt'
    
    dispenser_info = {}
    
    if map_file.exists():
        print(f"Reading map file: {map_file}")
        with open(map_file, 'r') as f:
            lines = f.readlines()
        
        # Parse map - look for dispenser positions
        # T=tomato dispenser, X=plate dispenser, B=cuttingboard, M=counter, D=delivery
        # Convert from 0-indexed map coordinates to 1-indexed coordinates used in actions.csv
        for y, line in enumerate(lines):
            line = line.strip()
            for x, char in enumerate(line):
                # Convert to 1-indexed coordinates
                coord_x, coord_y = x + 1, y + 1
                
                if char == 'X':  # Plate dispenser
                    dispenser_info[(coord_x, coord_y)] = 'plate'
                    print(f"  Plate dispenser at ({coord_x}, {coord_y})")
                elif char == 'T':  # Tomato dispenser  
                    dispenser_info[(coord_x, coord_y)] = 'tomato'
                    print(f"  Tomato dispenser at ({coord_x}, {coord_y})")
                elif char == 'P':  # Pumpkin dispenser (if any)
                    dispenser_info[(coord_x, coord_y)] = 'pumpkin'
                    print(f"  Pumpkin dispenser at ({coord_x}, {coord_y})")
                elif char == 'C':  # Cabbage dispenser (if any)
                    dispenser_info[(coord_x, coord_y)] = 'cabbage'
                    print(f"  Cabbage dispenser at ({coord_x}, {coord_y})")
    else:
        print(f"Map file not found: {map_file}")
        # Fallback: From actual map layout WBBWTTXW
        # Convert from 0-indexed map to 1-indexed coordinates used in actions.csv
        # X at position (6,0) -> (7,1), T at (4,0) -> (5,1) and (5,0) -> (6,1), also X at (7,1) -> (8,2)
        dispenser_info = {
            (5, 1): 'tomato',    # T at map position (4,0) 
            (6, 1): 'tomato',    # T at map position (5,0)
            (7, 1): 'plate',     # X at map position (6,0)
            (8, 2): 'plate'      # X at map position (7,1)
        }
        print("Using fallback dispenser info (1-indexed coordinates):")
        for pos, item_type in dispenser_info.items():
            print(f"  {item_type} dispenser at {pos}")
    
    return dispenser_info

def is_near_target(agent_x, agent_y, target_x, target_y, max_distance=1.5):
    """Check if agent is near the target tile (within max_distance tiles)
    
    Clickable tiles cannot be accessed diagonally, so we only allow orthogonal access.
    This means Manhattan distance of 1 (adjacent horizontally or vertically).
    """
    # Convert to integers to handle any floating point precision issues
    agent_x_int = int(round(agent_x))
    agent_y_int = int(round(agent_y))
    target_x_int = int(round(target_x))
    target_y_int = int(round(target_y))
    
    # Calculate Manhattan distance (no diagonal access)
    manhattan_distance = abs(agent_x_int - target_x_int) + abs(agent_y_int - target_y_int)
    return manhattan_distance <= 1

def find_nearby_tiles(agent_x, agent_y, actions_df, tile_types):
    """Find tiles of specific types that are orthogonally adjacent to the agent"""
    nearby_tiles = []
    
    # Check orthogonally adjacent positions (no diagonals)
    adjacent_positions = [
        (agent_x + 1, agent_y),  # Right
        (agent_x - 1, agent_y),  # Left
        (agent_x, agent_y + 1),  # Up
        (agent_x, agent_y - 1)   # Down
    ]
    
    for pos_x, pos_y in adjacent_positions:
        # Find actions that target this position with the specified tile types
        # Filter out actions with NaN coordinates (e.g., do_nothing actions)
        matching_actions = actions_df[
            (actions_df['target_tile_x'] == pos_x) & 
            (actions_df['target_tile_y'] == pos_y) & 
            (actions_df['target_tile_type'].isin(tile_types)) &
            (pd.notna(actions_df['target_tile_x'])) &
            (pd.notna(actions_df['target_tile_y']))
        ]
        
        for _, action in matching_actions.iterrows():
            nearby_tiles.append({
                'x': pos_x,
                'y': pos_y,
                'type': action['target_tile_type']
            })
    
    return nearby_tiles

def get_action_category(item_change_type, previous_item, current_item, target_tile_type):
    """Map item change to action category from actionMap"""
    
    # Define the actionMap
    actionMap = [
        'put down plate on counter',           # 0
        'pick up tomato from dispenser',       # 1
        'pick up plate from dispenser',        # 2
        'start cutting tomato',                # 3
        'pick up tomato_cut from cuttingboard', # 4
        'pick up tomato_salad from counter',   # 5
        'put down tomato on counter',          # 6
        'assemble salad',                      # 7
        'pick up tomato from counter',         # 8
        'deliver tomato_salad',                # 9
        'pick up plate from counter',          # 10
        'put down tomato_cut on counter',      # 11
        'pick up tomato_cut from counter',     # 12
        'put down tomato_salad on counter',    # 13
        'put down tomato on dispenser',        # 14 (destructive)
        'put down plate on dispenser',         # 15 (destructive)
        'put down plate on cuttingboard',      # 16
        'put down tomato_cut on dispenser',    # 17 (destructive)
        'put down tomato_salad on dispenser',  # 18 (destructive)
        'put down tomato on cuttingboard',     # 19
        'put down tomato_salad on cuttingboard' # 20 (destructive)
    ]
    
    # Normalize item names
    prev_item = previous_item if previous_item else 'None'
    curr_item = current_item if current_item else 'None'
    
    if item_change_type == "pickup":
        # Agent picked up an item
        if curr_item == 'plate':
            if target_tile_type == 'dispenser':
                return 2, actionMap[2]  # 'pick up plate from dispenser'
            elif target_tile_type == 'counter':
                return 10, actionMap[10]  # 'pick up plate from counter'
        
        elif curr_item == 'tomato':
            if target_tile_type == 'dispenser':
                return 1, actionMap[1]  # 'pick up tomato from dispenser'
            elif target_tile_type == 'counter':
                return 8, actionMap[8]  # 'pick up tomato from counter'
        
        elif curr_item == 'tomato_cut':
            if target_tile_type == 'cuttingboard':
                return 4, actionMap[4]  # 'pick up tomato_cut from cuttingboard'
            elif target_tile_type == 'counter':
                return 12, actionMap[12]  # 'pick up tomato_cut from counter'
        
        elif curr_item == 'tomato_salad':
            if target_tile_type == 'counter':
                return 5, actionMap[5]  # 'pick up tomato_salad from counter'
    
    elif item_change_type == "drop":
        # Agent dropped an item
        if prev_item == 'plate':
            if target_tile_type == 'counter':
                return 0, actionMap[0]  # 'put down plate on counter'
            elif target_tile_type == 'dispenser':
                return 15, actionMap[15]  # 'put down plate on dispenser' (destructive)
            elif target_tile_type == 'cuttingboard':
                return 16, actionMap[16]  # 'put down plate on cuttingboard'
        
        elif prev_item == 'tomato':
            if target_tile_type == 'counter':
                return 6, actionMap[6]  # 'put down tomato on counter'
            elif target_tile_type == 'dispenser':
                return 14, actionMap[14]  # 'put down tomato on dispenser' (destructive)
            elif target_tile_type == 'cuttingboard':
                return 19, actionMap[19]  # 'put down tomato on cuttingboard'
        
        elif prev_item == 'tomato_cut':
            if target_tile_type == 'counter':
                return 11, actionMap[11]  # 'put down tomato_cut on counter'
            elif target_tile_type == 'dispenser':
                return 17, actionMap[17]  # 'put down tomato_cut on dispenser' (destructive)
        
        elif prev_item == 'tomato_salad':
            if target_tile_type == 'counter':
                return 13, actionMap[13]  # 'put down tomato_salad on counter'
            elif target_tile_type == 'delivery':
                return 9, actionMap[9]  # 'deliver tomato_salad'
            elif target_tile_type == 'dispenser':
                return 18, actionMap[18]  # 'put down tomato_salad on dispenser' (destructive)
            elif target_tile_type == 'cuttingboard':
                return 20, actionMap[20]  # 'put down tomato_salad on cuttingboard' (destructive)
            
    # If no match found, return unknown
    return -1, f"UNKNOWN: {item_change_type} {prev_item} -> {curr_item} at {target_tile_type}"

def verify_cuttingboard_usage(simulation_df, agent_id, current_frame, agent_tile_x, agent_tile_y):
    """Look ahead 72 frames (3 seconds) to verify if agent actually used cuttingboard
    
    Returns:
        tuple: (is_cuttingboard, future_frame, future_item) where:
            - is_cuttingboard: True if agent is still on same tile with tomato_cut after 72 frames
            - future_frame: The frame where tomato_cut was detected (or None)
            - future_item: The item the agent has at that frame
    """
    target_frame = current_frame + 72  # 3 seconds later
    
    # Look for the agent in future frames (72 frames ahead, but also check a small window around it)
    search_frames = range(target_frame - 5, target_frame + 6)  # Small window to account for timing variations
    
    for check_frame in search_frames:
        future_data = simulation_df[
            (simulation_df['frame'] == check_frame) & 
            (simulation_df['agent_id'] == agent_id)
        ]
        
        if not future_data.empty:
            future_row = future_data.iloc[0]
            
            # Check for NaN values before converting to int
            if pd.isna(future_row['tile_x']) or pd.isna(future_row['tile_y']):
                print(f"      WARNING: Agent {agent_id} has NaN coordinates at frame {check_frame}, skipping cuttingboard verification")
                continue
                
            future_tile_x = int(future_row['tile_x'])
            future_tile_y = int(future_row['tile_y'])
            future_item = future_row['item'] if pd.notna(future_row['item']) else None
            
            # Check if agent is still on the same tile
            if future_tile_x == agent_tile_x and future_tile_y == agent_tile_y:
                # Check if agent now has tomato_cut
                if future_item == 'tomato_cut':
                    print(f"      Verified cuttingboard usage: Agent {agent_id} at frame {check_frame} still on tile ({agent_tile_x}, {agent_tile_y}) with tomato_cut")
                    return True, check_frame, future_item
                else:
                    print(f"      Agent {agent_id} at frame {check_frame} still on tile ({agent_tile_x}, {agent_tile_y}) but has item: {future_item} (not tomato_cut)")
            else:
                print(f"      Agent {agent_id} at frame {check_frame} moved to tile ({future_tile_x}, {future_tile_y}) - not cutting")
    
    print(f"      No cuttingboard verification found for agent {agent_id} around frame {target_frame}")
    return False, None, None

def analyze_meaningful_actions(actions_df, simulation_df, map_nr, output_dir=None):
    """Detect meaningful actions by finding item state changes and matching to actions.csv"""
    
    # Read the CSV files if they are paths, otherwise use the dataframes directly
    if isinstance(actions_df, (str, Path)):
        print("Reading CSV files...")
        actions_df = pd.read_csv(actions_df)
        simulation_df = pd.read_csv(simulation_df)
    
    print(f"Actions shape: {actions_df.shape}")
    print(f"Simulation shape: {simulation_df.shape}")
    
    # DEBUG: Check for NaN values in the data
    print("\n=== DEBUGGING NaN VALUES ===")
    print("Simulation DataFrame columns:", simulation_df.columns.tolist())
    print("Actions DataFrame columns:", actions_df.columns.tolist())
    
    # Check for NaN values in key columns
    if 'tile_x' in simulation_df.columns:
        nan_tile_x = simulation_df['tile_x'].isna().sum()
        print(f"NaN values in simulation tile_x: {nan_tile_x}")
        if nan_tile_x > 0:
            print("Sample rows with NaN tile_x:")
            print(simulation_df[simulation_df['tile_x'].isna()][['frame', 'agent_id', 'tile_x', 'tile_y', 'x', 'y', 'item']].head())
    
    if 'tile_y' in simulation_df.columns:
        nan_tile_y = simulation_df['tile_y'].isna().sum()
        print(f"NaN values in simulation tile_y: {nan_tile_y}")
        if nan_tile_y > 0:
            print("Sample rows with NaN tile_y:")
            print(simulation_df[simulation_df['tile_y'].isna()][['frame', 'agent_id', 'tile_x', 'tile_y', 'x', 'y', 'item']].head())
    
    # Check data types
    print("\nSimulation DataFrame dtypes:")
    print(simulation_df.dtypes)
    
    # Check for any completely empty rows or weird data
    print(f"\nTotal simulation rows: {len(simulation_df)}")
    print(f"Rows with NaN in tile_x OR tile_y: {simulation_df[['tile_x', 'tile_y']].isna().any(axis=1).sum()}")
    print(f"Unique agent_ids: {simulation_df['agent_id'].unique()}")
    print(f"Frame range: {simulation_df['frame'].min()} to {simulation_df['frame'].max()}")
    
    # Check for NaN values in actions DataFrame
    if 'target_tile_x' in actions_df.columns:
        nan_actions_x = actions_df['target_tile_x'].isna().sum()
        print(f"NaN values in actions target_tile_x: {nan_actions_x}")
        if nan_actions_x > 0:
            print("Sample actions with NaN target_tile_x (likely do_nothing actions):")
            nan_actions = actions_df[actions_df['target_tile_x'].isna()]
            print(nan_actions[['agent_id', 'action_id', 'action_number', 'action_type', 'target_tile_x', 'target_tile_y', 'target_tile_type']].head())
            print(f"Total actions with NaN coordinates: {len(nan_actions)} out of {len(actions_df)} ({100*len(nan_actions)/len(actions_df):.1f}%)")
    
    print("=== END NaN DEBUGGING ===\n")
    
    # Load map information
    dispenser_info = load_map_info(map_nr)
    
    # Sort by agent_id and action_id (chronological order)
    actions_df = actions_df.sort_values(['agent_id', 'action_id']).reset_index(drop=True)
    
    # Sort simulation by frame
    simulation_df = simulation_df.sort_values('frame').reset_index(drop=True)
    
    meaningful_actions = []
    
    # Track states for each agent
    agent_states = {}
    for agent_id in actions_df['agent_id'].unique():
        agent_states[agent_id] = {
            'current_action_idx': 0,
            'previous_item': None,
            'actions': actions_df[actions_df['agent_id'] == agent_id].copy().reset_index(drop=True),
            'skip_frames': set()  # Frames to skip processing (already handled in compound actions)
        }
        print(f"Agent {agent_id} has {len(agent_states[agent_id]['actions'])} actions")
    
    # Track counter states - what items are on each counter position (shared between all agents)
    # Key: (x, y) position, Value: item type on that counter
    counter_states = {}
    
    # Process all agents simultaneously, frame by frame
    for frame in sorted(simulation_df['frame'].unique()):
        frame_data = simulation_df[simulation_df['frame'] == frame]
        
        # Normalize counter_states keys to integers to avoid numpy float comparison issues
        normalized_counter_states = {}
        for pos, item in counter_states.items():
            normalized_pos = (int(pos[0]), int(pos[1]))
            normalized_counter_states[normalized_pos] = item
        counter_states = normalized_counter_states
        
        print(f"\n=== Processing Frame {frame} ===")
        print(f"    Current counter states: {counter_states}")
        
        # Process each agent in this frame
        for _, sim_row in frame_data.iterrows():
            agent_id = sim_row['agent_id']
            current_item = sim_row['item'] if pd.notna(sim_row['item']) else None
            previous_item = agent_states[agent_id]['previous_item']
            
            # DEBUG: Check this specific row for NaN values
            if pd.isna(sim_row['tile_x']) or pd.isna(sim_row['tile_y']):
                print(f"      DEBUG: Found NaN coordinates for agent {agent_id} at frame {frame}")
                print(f"        Full row data: {sim_row.to_dict()}")
                print(f"        tile_x: {sim_row['tile_x']} (type: {type(sim_row['tile_x'])})")
                print(f"        tile_y: {sim_row['tile_y']} (type: {type(sim_row['tile_y'])})")
                print(f"        x: {sim_row['x']} (type: {type(sim_row['x'])})")
                print(f"        y: {sim_row['y']} (type: {type(sim_row['y'])})")
                print(f"        item: {sim_row['item']}")
                
                # POTENTIAL FIX: Check if x,y coordinates exist and could be used
                if pd.notna(sim_row['x']) and pd.notna(sim_row['y']):
                    # Convert continuous coordinates to tile coordinates
                    # Assuming 32-pixel tiles (common in games)
                    converted_tile_x = int(sim_row['x'] // 32)
                    converted_tile_y = int(sim_row['y'] // 32)
                    print(f"        ATTEMPTING FIX: Converting x={sim_row['x']}, y={sim_row['y']} to tiles ({converted_tile_x}, {converted_tile_y})")
                    
                    # Use converted coordinates
                    agent_tile_x = converted_tile_x
                    agent_tile_y = converted_tile_y
                else:
                    print(f"      WARNING: Agent {agent_id} has NaN coordinates at frame {frame}, skipping")
                    continue
            else:
                agent_tile_x = int(sim_row['tile_x'])
                agent_tile_y = int(sim_row['tile_y'])
            
            # Detect item state change
            item_changed = False
            change_type = ""
            
            if previous_item != current_item:
                if previous_item is None and current_item is not None:
                    item_changed = True
                    change_type = "pickup"
                elif previous_item is not None and current_item is None:
                    item_changed = True
                    change_type = "drop"
                elif previous_item != current_item:
                    item_changed = True
                    change_type = "change"
            
            print(f"  Agent {agent_id}: {previous_item} -> {current_item} (change_type: {change_type if item_changed else 'no_change'}) at tile ({agent_tile_x}, {agent_tile_y})")
            
            if item_changed:
                print(f"      Item changed detected: {change_type}")
                
                # Find if agent is near any counter
                nearby_counters = find_nearby_tiles(agent_tile_x, agent_tile_y, 
                                                   agent_states[agent_id]['actions'], ['counter'])
                
                # Check for salad assembly BEFORE updating counter states (drop action or change to tomato_salad)
                is_salad_assembly = False
                if nearby_counters:
                    counter_pos = (int(nearby_counters[0]['x']), int(nearby_counters[0]['y']))
                    
                    # Drop action that completes a salad
                    if change_type == "drop" and previous_item in ['plate', 'tomato_cut']:
                        if previous_item == 'plate' and counter_pos in counter_states and counter_states[counter_pos] == 'tomato_cut':
                            is_salad_assembly = True
                        elif previous_item == 'tomato_cut' and counter_pos in counter_states and counter_states[counter_pos] == 'plate':
                            is_salad_assembly = True
                                
                # Get current action index for this agent
                current_action_idx = agent_states[agent_id]['current_action_idx']
                agent_actions = agent_states[agent_id]['actions']
                
                # Process salad assembly action if detected
                if is_salad_assembly:
                    print(f"      Processing as salad assembly")
                    # Find the action that matches this position
                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]
                        
                        # Skip actions with NaN coordinates (e.g., do_nothing actions)
                        if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                            continue
                            
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                               action['target_tile_x'], action['target_tile_y'])
                        
                        if is_near and action['target_tile_type'] == 'counter':
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            print(f"      Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                            break
                    
                    if matched_action is not None:
                        # This is a salad assembly action - the agent drops an item that combines with what's on the counter
                        # The actual item change is drop (previous_item -> None), but semantically it's assembly
                        # Use action category 7 for "assemble salad"
                        assembly_category_id = 7
                        assembly_category_name = "assemble salad"
                        
                        agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                        
                        assembly_item_change_type = "drop"
                        assembly_current_item = None  # Agent's hands are empty after dropping
                        
                        assembly_action = {
                            'frame': sim_row['frame'],
                            'agent_id': agent_id,
                            'action_id': matched_action['action_id'],
                            'action_number': matched_action['action_number'],
                            'action_type': matched_action['action_type'],
                            'target_tile_type': matched_action['target_tile_type'],
                            'target_tile_x': matched_action['target_tile_x'],
                            'target_tile_y': matched_action['target_tile_y'],
                            'agent_tile_x': agent_tile_x,
                            'agent_tile_y': agent_tile_y,
                            'item_change_type': assembly_item_change_type,
                            'previous_item': previous_item,
                            'current_item': assembly_current_item,
                            'agent_x': sim_row['x'],
                            'agent_y': sim_row['y'],
                            'action_category_id': assembly_category_id,
                            'action_category_name': assembly_category_name,
                            'compound_action_part': 0  # Single assembly action
                        }
                        meaningful_actions.append(assembly_action)
                        print(f"      Assembly: {assembly_category_name}")
                        
                        # Update counter states after salad assembly - the result is tomato_salad
                        counter_pos = (int(matched_action['target_tile_x']), int(matched_action['target_tile_y']))
                        counter_states[counter_pos] = 'tomato_salad'
                        print(f"      Updated counter {counter_pos} with tomato_salad after assembly")
                    else:
                        print(f"      Could not match salad assembly action")
                
                # Check if this is a compound action (item change at dispenser)
                elif change_type == "change" and previous_item is not None and current_item is not None and not is_salad_assembly:
                    
                    print(f"      Checking for compound action or transformation: {previous_item} -> {current_item}")
                    print(f"      Current action index: {current_action_idx}/{len(agent_actions)}")
                    check_counter = True

                    if current_item in ['plate', 'tomato']:
                        # First try dispensers (traditional compound action)
                        matched_action = None
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (e.g., do_nothing actions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                                
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])

                            print(f"        Checking dispenser action [{check_idx}]: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}] - Near: {is_near}")

                            # For traditional compound actions, match dispensers
                            if is_near and action['target_tile_type'] == 'dispenser':
                                # Verify that the dispenser is of the correct type for the current item
                                dispenser_pos = (action['target_tile_x'], action['target_tile_y'])
                                expected_dispenser_type = dispenser_info.get(dispenser_pos)
                                
                                # Check if dispenser type matches the current item being picked up
                                if expected_dispenser_type == current_item:
                                    matched_action = action.copy()
                                    matched_action['matched_action_idx'] = check_idx
                                    print(f"      Matched to dispenser action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}] - Dispenser type: {expected_dispenser_type}")
                                    break
                                else:
                                    print(f"        Dispenser type mismatch: expected {expected_dispenser_type}, got {current_item} at ({action['target_tile_x']}, {action['target_tile_y']})")
                            
                        if matched_action is not None:
                            check_counter = False
                            print(f"      Processing as compound action at dispenser")
                            # Update action index AFTER processing
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                            # Traditional compound action (put down old item + pick up new item)
                            # 1. Put down the previous item (destructive action)
                            put_down_category_id, put_down_category_name = get_action_category(
                                "drop", previous_item, None, matched_action['target_tile_type']
                            )

                            put_down_action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'agent_tile_y': agent_tile_y,
                                'item_change_type': "drop",
                                'previous_item': previous_item,
                                'current_item': None,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': put_down_category_id,
                                'action_category_name': put_down_category_name,
                                'compound_action_part': 1  # First part of compound action
                            }
                            meaningful_actions.append(put_down_action)
                            print(f"      Part 1 - Put down: {put_down_category_name}")
                            
                            # Update counter states if dropping on counter
                            if matched_action['target_tile_type'] == 'counter':
                                counter_pos = (int(matched_action['target_tile_x']), int(matched_action['target_tile_y']))
                                counter_states[counter_pos] = previous_item
                                print(f"      Updated counter {counter_pos} with {previous_item}")

                            # 2. Pick up the new item
                            pick_up_category_id, pick_up_category_name = get_action_category(
                                "pickup", None, current_item, matched_action['target_tile_type']
                            )

                            pick_up_action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'agent_tile_y': agent_tile_y,
                                'item_change_type': "pickup",
                                'previous_item': None,
                                'current_item': current_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': pick_up_category_id,
                                'action_category_name': pick_up_category_name,
                                'compound_action_part': 2  # Second part of compound action
                            }
                            meaningful_actions.append(pick_up_action)
                            print(f"      Part 2 - Pick up: {pick_up_category_name}")

                        else:
                            print(f"      No dispenser found - checking for counter-based transformation")

                    if check_counter:
                        # Look for counter actions (like putting down tomato and picking up plate from same counter)
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (e.g., do_nothing actions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                                
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])

                            print(f"        Checking counter action [{check_idx}]: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}] - Near: {is_near}")

                            if is_near and action['target_tile_type'] == 'counter':
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to counter action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                                break
                            
                        if matched_action is not None:
                            print(f"      Processing as compound action at counter")
                            # Update action index AFTER processing
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                            
                            # Store what was originally on the counter before the compound action
                            counter_pos = (int(matched_action['target_tile_x']), int(matched_action['target_tile_y']))
                            original_counter_item = counter_states.get(counter_pos, None)
                            
                            # Compound action at counter: put down old item + pick up new item
                            # 1. Put down the previous item
                            put_down_category_id, put_down_category_name = get_action_category(
                                "drop", previous_item, None, matched_action['target_tile_type']
                            )

                            put_down_action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'agent_tile_y': agent_tile_y,
                                'item_change_type': "drop",
                                'previous_item': previous_item,
                                'current_item': None,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': put_down_category_id,
                                'action_category_name': put_down_category_name,
                                'compound_action_part': 1  # First part of compound action
                            }
                            meaningful_actions.append(put_down_action)
                            print(f"      Counter Part 1 - Put down: {put_down_category_name}")
                            
                            # DON'T update counter state yet - we need to pick up what was originally there first
                            
                            # 2. Pick up the new item (what was originally on the counter)
                            pick_up_category_id, pick_up_category_name = get_action_category(
                                "pickup", None, current_item, matched_action['target_tile_type']
                            )

                            pick_up_action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'agent_tile_y': agent_tile_y,
                                'item_change_type': "pickup",
                                'previous_item': None,
                                'current_item': current_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': pick_up_category_id,
                                'action_category_name': pick_up_category_name,
                                'compound_action_part': 2  # Second part of compound action
                            }
                            meaningful_actions.append(pick_up_action)
                            print(f"      Counter Part 2 - Pick up: {pick_up_category_name}")
                            
                            # NOW update counter state: what the agent put down replaces what was there
                            counter_states[counter_pos] = previous_item
                            print(f"      Updated counter {counter_pos} with {previous_item} (replacing {original_counter_item})")
                        else:
                            print(f"      Could not match transformation to any action")
                
                # Handle cuttingboard mechanics separately
                elif change_type == "drop" and previous_item == 'tomato' and current_item is None:
                    print(f"      Checking if this is cuttingboard tomato drop")
                    # Check if this is putting tomato on cuttingboard
                    nearby_cuttingboards = find_nearby_tiles(agent_tile_x, agent_tile_y, agent_actions, ['cuttingboard'])
                    
                    if nearby_cuttingboards:
                        print(f"      Found nearby cuttingboard - verifying actual usage by looking ahead 72 frames")
                        
                        # Verify this is actually cuttingboard usage by looking ahead 72 frames
                        is_actual_cuttingboard, future_frame, future_item = verify_cuttingboard_usage(
                            simulation_df, agent_id, sim_row['frame'], agent_tile_x, agent_tile_y
                        )
                        
                        if is_actual_cuttingboard:
                            print(f"      Confirmed cuttingboard usage - processing complete cuttingboard sequence")
                            # This is confirmed cuttingboard usage - process all three steps at once
                            matched_action = None
                            for check_idx in range(current_action_idx, len(agent_actions)):
                                action = agent_actions.iloc[check_idx]
                                
                                # Skip actions with NaN coordinates (e.g., do_nothing actions)
                                if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                    continue
                                    
                                is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                       action['target_tile_x'], action['target_tile_y'])
                                
                                if is_near and action['target_tile_type'] == 'cuttingboard':
                                    matched_action = action.copy()
                                    matched_action['matched_action_idx'] = check_idx
                                    break
                            
                            if matched_action is not None:
                                # Step 1: Put down tomato on cuttingboard
                                put_down_category_id, put_down_category_name = get_action_category(
                                    "drop", previous_item, None, matched_action['target_tile_type']
                                )
                                
                                put_down_action = {
                                    'frame': sim_row['frame'],
                                    'agent_id': agent_id,
                                    'action_id': matched_action['action_id'],
                                    'action_number': matched_action['action_number'],
                                    'action_type': matched_action['action_type'],
                                    'target_tile_type': matched_action['target_tile_type'],
                                    'target_tile_x': matched_action['target_tile_x'],
                                    'target_tile_y': matched_action['target_tile_y'],
                                    'agent_tile_x': agent_tile_x,
                                    'agent_tile_y': agent_tile_y,
                                    'item_change_type': "drop",
                                    'previous_item': previous_item,
                                    'current_item': None,
                                    'agent_x': sim_row['x'],
                                    'agent_y': sim_row['y'],
                                    'action_category_id': put_down_category_id,
                                    'action_category_name': put_down_category_name,
                                    'compound_action_part': 1  # First part of cuttingboard sequence
                                }
                                meaningful_actions.append(put_down_action)
                                print(f"      Cuttingboard Step 1: {put_down_category_name}")
                                
                                # Step 2: Start cutting tomato (intermediate frame)
                                start_cutting_action = {
                                    'frame': sim_row['frame'],  # Use same frame as step 1 for logical grouping
                                    'agent_id': agent_id,
                                    'action_id': matched_action['action_id'],
                                    'action_number': matched_action['action_number'],
                                    'action_type': matched_action['action_type'],
                                    'target_tile_type': matched_action['target_tile_type'],
                                    'target_tile_x': matched_action['target_tile_x'],
                                    'target_tile_y': matched_action['target_tile_y'],
                                    'agent_tile_x': agent_tile_x,
                                    'agent_tile_y': agent_tile_y,
                                    'item_change_type': "cutting",  # Special type for cutting process
                                    'previous_item': None,
                                    'current_item': None,
                                    'agent_x': sim_row['x'],
                                    'agent_y': sim_row['y'],
                                    'action_category_id': 3,  # start cutting tomato
                                    'action_category_name': 'start cutting tomato',
                                    'compound_action_part': 2  # Second part of cuttingboard sequence
                                }
                                meaningful_actions.append(start_cutting_action)
                                print(f"      Cuttingboard Step 2: start cutting tomato")
                                
                                # Step 3: Pick up tomato_cut from cuttingboard (use future frame)
                                pick_up_category_id, pick_up_category_name = get_action_category(
                                    "pickup", None, future_item, matched_action['target_tile_type']
                                )
                                
                                pick_up_action = {
                                    'frame': future_frame,  # Use the future frame where pickup actually happens
                                    'agent_id': agent_id,
                                    'action_id': matched_action['action_id'],
                                    'action_number': matched_action['action_number'],
                                    'action_type': matched_action['action_type'],
                                    'target_tile_type': matched_action['target_tile_type'],
                                    'target_tile_x': matched_action['target_tile_x'],
                                    'target_tile_y': matched_action['target_tile_y'],
                                    'agent_tile_x': agent_tile_x,
                                    'agent_tile_y': agent_tile_y,
                                    'item_change_type': "pickup",
                                    'previous_item': None,
                                    'current_item': future_item,
                                    'agent_x': sim_row['x'],
                                    'agent_y': sim_row['y'],
                                    'action_category_id': pick_up_category_id,
                                    'action_category_name': pick_up_category_name,
                                    'compound_action_part': 3  # Third part of cuttingboard sequence
                                }
                                meaningful_actions.append(pick_up_action)
                                print(f"      Cuttingboard Step 3: {pick_up_category_name}")
                                
                                # Increment action index after processing complete sequence
                                agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                                
                                # Mark the future frame to skip processing the tomato_cut pickup later
                                if agent_id not in agent_states:
                                    agent_states[agent_id] = {}
                                if 'skip_frames' not in agent_states[agent_id]:
                                    agent_states[agent_id]['skip_frames'] = set()
                                agent_states[agent_id]['skip_frames'].add(future_frame)
                                
                            else:
                                print(f"      Could not match cuttingboard action")
                        else:
                            print(f"      Not actual cuttingboard usage - treating as regular counter drop")
                            # This is just a regular drop on what appears to be a cuttingboard
                            # but is actually being used as a counter - process as regular drop
                    
                    # Process as regular drop (either no cuttingboard nearby, or failed cuttingboard verification)
                    if not (nearby_cuttingboards and len(meaningful_actions) > 0 and 
                           meaningful_actions[-1]['compound_action_part'] == 3):  # Check if cuttingboard sequence was completed
                        print(f"      Processing as regular drop action")
                        matched_action = None
                        
                        # Start looking from current action index forward (cannot be previous actions)
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (e.g., do_nothing actions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                            
                            # Check if agent is near the target tile of this action
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])
                            
                            if not is_near:
                                continue
                            
                            print(f"        Checking action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                            
                            # Check if the action type matches the item change
                            action_matches = False
                            
                            # Agent dropped item, should be near delivery, counter, or cuttingboard
                            action_matches = action['target_tile_type'] in ['delivery', 'counter', 'cuttingboard']
                            
                            print(f"        Action matches: {action_matches}")
                            
                            if action_matches:
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                                break
                        
                        if matched_action is not None:
                            # Get the action category
                            action_category_id, action_category_name = get_action_category(
                                change_type, previous_item, current_item, matched_action['target_tile_type']
                            )
                            
                            # Save this meaningful action
                            meaningful_action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'agent_tile_y': agent_tile_y,
                                'item_change_type': change_type,
                                'previous_item': previous_item,
                                'current_item': current_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': action_category_id,
                                'action_category_name': action_category_name,
                                'compound_action_part': 0  # Single action (not compound)
                            }
                            meaningful_actions.append(meaningful_action)
                            print(f"      Category: {action_category_name}")
                            
                            # Update counter states for drop/pickup actions on counters
                            if change_type == "drop" and matched_action['target_tile_type'] == 'counter':
                                counter_pos = (matched_action['target_tile_x'], matched_action['target_tile_y'])
                                counter_states[counter_pos] = previous_item
                                print(f"      Updated counter {counter_pos} with {previous_item}")
                            elif change_type == "pickup" and matched_action['target_tile_type'] == 'counter':
                                counter_pos = (matched_action['target_tile_x'], matched_action['target_tile_y'])
                                if counter_pos in counter_states:
                                    del counter_states[counter_pos]
                                    print(f"      Removed item from counter {counter_pos}")
                            
                            # Only increment action index AFTER processing the action
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                        else:
                            print(f"      Could not match tomato drop to any action")
                    
                elif change_type == "pickup" and previous_item is None and current_item == 'tomato_cut':
                    # Check if this pickup should be skipped (already processed as part of cuttingboard sequence)
                    if (agent_id in agent_states and 
                        'skip_frames' in agent_states[agent_id] and 
                        sim_row['frame'] in agent_states[agent_id]['skip_frames']):
                        print(f"      Skipping tomato_cut pickup at frame {sim_row['frame']} - already processed as cuttingboard sequence")
                        agent_states[agent_id]['skip_frames'].remove(sim_row['frame'])
                        # Don't process this pickup - it was already handled in the cuttingboard sequence
                    else:
                        # This is a regular tomato_cut pickup (from counter or other location)
                        print(f"      Processing regular tomato_cut pickup")
                        # Process as regular pickup action
                        matched_action = None
                        
                        # Start looking from current action index forward (cannot be previous actions)
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (e.g., do_nothing actions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                            
                            # Check if agent is near the target tile of this action
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])
                            
                            if not is_near:
                                continue
                            
                            print(f"        Checking action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                            
                            # Check if the action type matches the item change (pickup from counter or cuttingboard)
                            action_matches = action['target_tile_type'] in ['counter', 'cuttingboard']
                            
                            print(f"        Action matches: {action_matches}")
                            
                            if action_matches:
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                                break
                        
                        if matched_action is not None:
                            # Get the action category
                            action_category_id, action_category_name = get_action_category(
                                change_type, previous_item, current_item, matched_action['target_tile_type']
                            )
                            
                            # Save this meaningful action
                            meaningful_action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'agent_tile_y': agent_tile_y,
                                'item_change_type': change_type,
                                'previous_item': previous_item,
                                'current_item': current_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': action_category_id,
                                'action_category_name': action_category_name,
                                'compound_action_part': 0  # Single action (not compound)
                            }
                            meaningful_actions.append(meaningful_action)
                            print(f"      Category: {action_category_name}")
                            
                            # Update counter states for pickup actions on counters
                            if change_type == "pickup" and matched_action['target_tile_type'] == 'counter':
                                counter_pos = (matched_action['target_tile_x'], matched_action['target_tile_y'])
                                if counter_pos in counter_states:
                                    del counter_states[counter_pos]
                                    print(f"      Removed item from counter {counter_pos}")
                            
                            # Only increment action index AFTER processing the action
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                        else:
                            print(f"      Could not match tomato_cut pickup to any action")
                else:
                    # Regular single action (pickup, drop, or transformation) - but skip if already handled as salad assembly or cutting
                    if not is_salad_assembly:
                        print(f"      Looking for regular action for {change_type}: {previous_item} -> {current_item}")
                        print(f"      Current action index: {current_action_idx}/{len(agent_actions)}")
                        matched_action = None
                        
                        # Start looking from current action index forward (cannot be previous actions)
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (e.g., do_nothing actions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                            
                            # Check if agent is near the target tile of this action
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])
                            
                            print(f"        Checking action [{check_idx}]: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}] - Near: {is_near}")
                            
                            if not is_near:
                                continue
                            
                            # Check if the action type matches the item change
                            action_matches = False
                            
                            if change_type == "pickup":
                                # Agent picked up item, should be near a dispenser or counter
                                action_matches = action['target_tile_type'] in ['dispenser', 'counter']
                                
                            elif change_type == "drop":
                                # Agent dropped item, should be near delivery, counter, or cuttingboard
                                action_matches = action['target_tile_type'] in ['delivery', 'counter', 'cuttingboard']
                                
                            elif change_type == "change":
                                # Item transformation, should be near counter
                                action_matches = action['target_tile_type'] in ['counter']
                            
                            print(f"        Action matches: {action_matches}")
                            
                            if action_matches:
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                                break
                        
                        if matched_action is not None:
                            # Get the action category
                            action_category_id, action_category_name = get_action_category(
                                change_type, previous_item, current_item, matched_action['target_tile_type']
                            )
                            
                            # Save this meaningful action
                            meaningful_action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'agent_tile_y': agent_tile_y,
                                'item_change_type': change_type,
                                'previous_item': previous_item,
                                'current_item': current_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': action_category_id,
                                'action_category_name': action_category_name,
                                'compound_action_part': 0  # Single action (not compound)
                            }
                            meaningful_actions.append(meaningful_action)
                            print(f"      Category: {action_category_name}")
                            
                            # Update counter states for drop/pickup actions on counters
                            if change_type == "drop" and matched_action['target_tile_type'] == 'counter':
                                counter_pos = (matched_action['target_tile_x'], matched_action['target_tile_y'])
                                counter_states[counter_pos] = previous_item
                                print(f"      Updated counter {counter_pos} with {previous_item}")
                            elif change_type == "pickup" and matched_action['target_tile_type'] == 'counter':
                                counter_pos = (matched_action['target_tile_x'], matched_action['target_tile_y'])
                                if counter_pos in counter_states:
                                    del counter_states[counter_pos]
                                    print(f"      Removed item from counter {counter_pos}")
                            
                            # Only increment action index AFTER processing the action
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                        else:
                            print(f"      Could not match item change to any action")
            
            # No longer need to check for cuttingboard step 2 separately - it's handled in the complete sequence
            
            # Check if item change was not processed
            elif item_changed:
                print(f"      WARNING: Item change not processed: {change_type} {previous_item} -> {current_item}")
            
            # Update the previous item for this agent
            agent_states[agent_id]['previous_item'] = current_item
    
    # Convert to DataFrame
    meaningful_df = pd.DataFrame(meaningful_actions)
    
    if not meaningful_df.empty:
        print(f"\nFound {len(meaningful_df)} meaningful actions:")
        print(meaningful_df[['frame', 'agent_id', 'action_number', 'target_tile_type', 'item_change_type', 'previous_item', 'current_item', 'action_category_name', 'compound_action_part']])
        
        # Save the meaningful actions if output directory is provided
        if output_dir is not None:
            output_path = Path(output_dir) / 'meaningful_actions.csv'
            meaningful_df.to_csv(output_path, index=False)
            print(f"\nSaved meaningful actions to: {output_path}")
    else:
        print("\nNo meaningful actions found!")
    
    return meaningful_df


# Legacy function for backward compatibility when running as standalone script
def analyze_meaningful_actions_from_files(actions_csv_path, simulation_csv_path, map_nr, output_dir=None):
    """Analyze meaningful actions from CSV file paths (for backward compatibility)"""
    return analyze_meaningful_actions(actions_csv_path, simulation_csv_path, map_nr, output_dir)


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze meaningful actions from simulation data')
    parser.add_argument('--actions_csv', required=True, help='Path to actions.csv file')
    parser.add_argument('--simulation_csv', required=True, help='Path to simulation.csv file')
    parser.add_argument('--map', required=True, help='Map number or name')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run analysis
    actions_csv = Path(args.actions_csv)
    simulation_csv = Path(args.simulation_csv)
    map_nr = args.map
    output_dir = Path(args.output_dir)
    
    result = analyze_meaningful_actions(actions_csv, simulation_csv, map_nr, output_dir)
    print("\nAnalysis complete!")