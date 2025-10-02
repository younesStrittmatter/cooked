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
    """Check if agent is near the target tile (within max_distance tiles)"""
    distance = math.sqrt((agent_x - target_x)**2 + (agent_y - target_y)**2)
    return distance <= max_distance

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
    
    elif item_change_type == "change":
        # Item transformation
        if prev_item == 'tomato' and curr_item == 'tomato_cut':
            return 3, actionMap[3]  # 'start cutting tomato'
        elif prev_item == 'plate' and curr_item == 'tomato_salad':
            return 7, actionMap[7]  # 'assemble salad'
        elif prev_item == 'tomato_cut' and curr_item == 'tomato_salad':
            return 7, actionMap[7]  # 'assemble salad'
    
    # If no match found, return unknown
    return -1, f"UNKNOWN: {item_change_type} {prev_item} -> {curr_item} at {target_tile_type}"

def analyze_meaningful_actions(actions_df, simulation_df, map_nr, output_dir=None):
    """Detect meaningful actions by finding item state changes and matching to actions.csv"""
    
    # Read the CSV files if they are paths, otherwise use the dataframes directly
    if isinstance(actions_df, (str, Path)):
        print("Reading CSV files...")
        actions_df = pd.read_csv(actions_df)
        simulation_df = pd.read_csv(simulation_df)
    
    print(f"Actions shape: {actions_df.shape}")
    print(f"Simulation shape: {simulation_df.shape}")
    
    # Load map information
    dispenser_info = load_map_info(map_nr)
    
    # Sort by agent_id and action_id (chronological order)
    actions_df = actions_df.sort_values(['agent_id', 'action_id']).reset_index(drop=True)
    
    # Sort simulation by frame
    simulation_df = simulation_df.sort_values('frame').reset_index(drop=True)
    
    meaningful_actions = []
    
    # Process each agent separately
    for agent_id in actions_df['agent_id'].unique():
        print(f"\nAnalyzing agent: {agent_id}")
        
        # Get agent's data
        agent_actions = actions_df[actions_df['agent_id'] == agent_id].copy().reset_index(drop=True)
        agent_simulation = simulation_df[simulation_df['agent_id'] == agent_id].copy()
        
        if agent_simulation.empty:
            print(f"No simulation data for agent {agent_id}")
            continue
        
        print(f"Agent has {len(agent_actions)} actions and {len(agent_simulation)} simulation frames")
        
        # Track current action index for this agent
        current_action_idx = 0
        previous_item = None
        
        # Go through simulation frame by frame
        for sim_idx, sim_row in agent_simulation.iterrows():
            current_item = sim_row['item'] if pd.notna(sim_row['item']) else None
            
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
            
            if item_changed:
                print(f"  Frame {sim_row['frame']}: Item {change_type} - {previous_item} -> {current_item}")
                
                # Look at agent's position
                agent_tile_x = int(sim_row['tile_x'])
                agent_tile_y = int(sim_row['tile_y'])
                
                # Check if this is a compound action (item change at dispenser)
                # This means: put down old item + pick up new item
                if change_type == "change" and previous_item is not None and current_item is not None:
                    
                    # Find the action that matches this position
                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                               action['target_tile_x'], action['target_tile_y'])
                        if is_near and action['target_tile_type'] == 'dispenser':
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            current_action_idx = check_idx + 1
                            print(f"    Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                            break
                    
                    if matched_action is not None:
                        # Create TWO meaningful actions:
                        
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
                        print(f"    Part 1 - Put down: {put_down_category_name}")
                        
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
                        print(f"    Part 2 - Pick up: {pick_up_category_name}")
                    else:
                        print(f"    Could not match compound action to any action")
                
                else:
                    # Regular single action (pickup, drop, or transformation)
                    matched_action = None
                    
                    # Start looking from current action index forward (cannot be previous actions)
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]
                        
                        # Check if agent is near the target tile of this action
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                               action['target_tile_x'], action['target_tile_y'])
                        
                        if not is_near:
                            continue
                        
                        # Check if the action type matches the item change
                        action_matches = False
                        
                        if change_type == "pickup":
                            # Agent picked up item, should be near a dispenser or counter
                            action_matches = action['target_tile_type'] in ['dispenser', 'counter']
                            
                        elif change_type == "drop":
                            # Agent dropped item, should be near delivery or counter
                            action_matches = action['target_tile_type'] in ['delivery', 'counter']
                            
                        elif change_type == "change":
                            # Item transformation, should be near counter or cuttingboard
                            action_matches = action['target_tile_type'] in ['counter', 'cuttingboard']
                        
                        if action_matches:
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            current_action_idx = check_idx + 1  # Move to next action for future searches
                            print(f"    Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
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
                        print(f"    Category: {action_category_name}")
                    else:
                        print(f"    Could not match item change to any action")
            
            previous_item = current_item
    
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
    # This section is for running the script standalone with the original parameters
    # (kept for backward compatibility)
    
    CLUSTER='cuenca'  # Options: 'brigit', 'local', 'cuenca'
    
    training_map_nr = "baseline_division_of_labor"
    num_agents = 2
    intent_version = "v3.1"
    cooperative = 1
    game_version = "classic"
    training_id = "2025-09-19_14-40-19"
    checkpoint_number = 62500
    simulation_id = "2025_10_02-06_57_49"
    
    # paths
    if CLUSTER == 'brigit':
        local = '/mnt/lustre/home/samuloza'
    elif CLUSTER == 'cuenca':
        local = ''
    elif CLUSTER == 'local':
        local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
    else:
        raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")
    
    base_path = Path(f"{local}/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{training_map_nr}")
    base_path = base_path / (f"cooperative/Training_{training_id}" if cooperative else f"competitive/Training_{training_id}")
    base_dir = base_path / f"simulations_{checkpoint_number}" / f"simulation_{simulation_id}"
    
    # Run analysis
    actions_csv = base_dir / 'actions.csv'
    simulation_csv = base_dir / 'simulation.csv'
    
    result = analyze_meaningful_actions(actions_csv, simulation_csv, training_map_nr, base_dir)
    print("\nAnalysis complete!")