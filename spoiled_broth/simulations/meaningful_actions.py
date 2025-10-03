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
    # Calculate Manhattan distance (no diagonal access)
    manhattan_distance = abs(agent_x - target_x) + abs(agent_y - target_y)
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
        matching_actions = actions_df[
            (actions_df['target_tile_x'] == pos_x) & 
            (actions_df['target_tile_y'] == pos_y) & 
            (actions_df['target_tile_type'].isin(tile_types))
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
        
        # Track counter states - what items are on each counter position
        # Key: (x, y) position, Value: item type on that counter
        counter_states = {}
        
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
                
                # Find if agent is near any counter
                nearby_counters = find_nearby_tiles(agent_tile_x, agent_tile_y, agent_actions, ['counter'])
                
                # Check for salad assembly BEFORE updating counter states (special case: drop action that assembles salad)
                is_salad_assembly = False
                if change_type == "drop" and previous_item in ['plate', 'tomato_cut'] and nearby_counters:
                    counter_pos = (nearby_counters[0]['x'], nearby_counters[0]['y'])
                    
                    # Check if this drop action completes a salad
                    if previous_item == 'plate' and counter_pos in counter_states and counter_states[counter_pos] == 'tomato_cut':
                        is_salad_assembly = True
                        print(f"    Salad assembly detected: putting down plate on counter with tomato_cut")
                    elif previous_item == 'tomato_cut' and counter_pos in counter_states and counter_states[counter_pos] == 'plate':
                        is_salad_assembly = True
                        print(f"    Salad assembly detected: putting down tomato_cut on counter with plate")
                
                # Update counter states when items are put down or picked up (AFTER checking for salad assembly)
                if nearby_counters:
                    counter_pos = (nearby_counters[0]['x'], nearby_counters[0]['y'])
                    
                    if change_type == "drop":
                        if is_salad_assembly:
                            # When assembling salad, the counter now has tomato_salad
                            counter_states[counter_pos] = 'tomato_salad'
                            print(f"    Counter at {counter_pos} now has: tomato_salad (after assembly)")
                        else:
                            # Agent put down an item on counter
                            counter_states[counter_pos] = previous_item
                            print(f"    Counter at {counter_pos} now has: {previous_item}")
                    elif change_type == "pickup":
                        # Agent picked up an item from counter
                        if counter_pos in counter_states:
                            del counter_states[counter_pos]
                            print(f"    Counter at {counter_pos} is now empty")
                
                # Process salad assembly action if detected
                if is_salad_assembly:
                    # Find the action that matches this position
                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                               action['target_tile_x'], action['target_tile_y'])
                        
                        if is_near and action['target_tile_type'] == 'counter':
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            current_action_idx = check_idx + 1
                            print(f"    Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                            break
                    
                    if matched_action is not None:
                        # This is a salad assembly action - the agent drops an item that combines with what's on the counter
                        # The actual item change is drop (previous_item -> None), but semantically it's assembly
                        assembly_category_id, assembly_category_name = get_action_category(
                            "change", previous_item, "tomato_salad", matched_action['target_tile_type']
                        )
                        
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
                            'item_change_type': "change",  # Semantically this is assembly/transformation
                            'previous_item': previous_item,
                            'current_item': "tomato_salad",  # What gets created (even though agent doesn't hold it)
                            'agent_x': sim_row['x'],
                            'agent_y': sim_row['y'],
                            'action_category_id': assembly_category_id,
                            'action_category_name': assembly_category_name,
                            'compound_action_part': 0  # Single assembly action
                        }
                        meaningful_actions.append(assembly_action)
                        print(f"    Assembly: {assembly_category_name}")
                    else:
                        print(f"    Could not match salad assembly action")
                
                # Check if this is a compound action (item change at dispenser)
                # OR a transformation action (cutting tomato) - but skip if already handled as salad assembly
                elif change_type == "change" and previous_item is not None and current_item is not None and not is_salad_assembly:
                    
                    # Special logic for different item transformations
                    if current_item == 'tomato_cut':
                        # If ending with tomato_cut, determine if it came from cuttingboard or counter
                        nearby_tiles = find_nearby_tiles(agent_tile_x, agent_tile_y, agent_actions, ['cuttingboard', 'counter'])
                        
                        # If there's a cuttingboard nearby AND previous item was tomato, use cuttingboard
                        has_nearby_cuttingboard = any(tile['type'] == 'cuttingboard' for tile in nearby_tiles)
                        
                        if has_nearby_cuttingboard and previous_item == 'tomato':
                            transformation_tile_type = 'cuttingboard'
                            print(f"    Tomato cutting detected: tomato -> tomato_cut using cuttingboard")
                        else:
                            # No cuttingboard nearby or previous item wasn't tomato
                            transformation_tile_type = 'counter'
                            print(f"    Item exchange at counter: {previous_item} -> tomato_cut")
                    
                    else:
                        # Other item changes - look for dispensers (traditional compound action)
                        transformation_tile_type = 'dispenser'
                    
                    # Find the action that matches this position and tile type
                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                               action['target_tile_x'], action['target_tile_y'])
                        
                        # For transformations (tomato_cut/tomato_salad), match the determined tile type
                        # For traditional compound actions, match dispensers
                        if is_near and action['target_tile_type'] == transformation_tile_type:
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            current_action_idx = check_idx + 1
                            print(f"    Matched to action {check_idx}: {action['action_type']} -> ({action['target_tile_x']}, {action['target_tile_y']}) [{action['target_tile_type']}]")
                            break
                    
                    if matched_action is not None:
                        # Handle transformations vs compound actions differently
                        if transformation_tile_type in ['cuttingboard', 'counter'] and (
                            (current_item == 'tomato_cut' and previous_item == 'tomato')  # Cutting tomato
                        ):
                            # This is a transformation action (cutting or assembling)
                            transformation_category_id, transformation_category_name = get_action_category(
                                "change", previous_item, current_item, matched_action['target_tile_type']
                            )
                            
                            transformation_action = {
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
                                'item_change_type': "change",
                                'previous_item': previous_item,
                                'current_item': current_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': transformation_category_id,
                                'action_category_name': transformation_category_name,
                                'compound_action_part': 0  # Single transformation action
                            }
                            meaningful_actions.append(transformation_action)
                            print(f"    Transformation: {transformation_category_name}")
                        
                        else:
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
                    # Regular single action (pickup, drop, or transformation) - but skip if already handled as salad assembly
                    if not is_salad_assembly:
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