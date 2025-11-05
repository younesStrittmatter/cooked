# USE: nohup python spoiled_broth/simulations/meaningful_actions.py --actions_csv spoiled_broth/simulations/actions.csv --simulation_csv spoiled_broth/simulations/simulation.csv --map 1 --output_dir spoiled_broth/simulations/ > log_meaningful_actions.out 2>&1 &

"""
Meaningful actions analysis for simulation runs.

This module provides functionality to detect meaningful actions by finding 
item state changes and matching them to actions.csv data.

Author: Samuel Lozano
"""

import pandas as pd
import numpy as np
import math
import re
from pathlib import Path

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

# Create a mapping of new action types to our categories
def get_action_category(action_type, item_change_type, previous_agent_item, current_agent_item, target_tile_type):
    """Map action type to action category"""
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
        'put down plate on cuttingboard',      # 16 (destructive)
        'put down tomato_cut on dispenser',    # 17 (destructive)
        'put down tomato_salad on dispenser',  # 18 (destructive)
        'put down tomato on cuttingboard',     # 19
        'put down tomato_salad on cuttingboard' # 20 (destructive)
    ]
    
    # Map action types that include coordinates
    if action_type in ['pick_up_plate_from_dispenser']:
        return 2, actionMap[2]  # 'pick up plate from dispenser'
    elif action_type in ['pick_up_tomato_from_dispenser']:
        return 1, actionMap[1]  # 'pick up tomato from dispenser'
    elif action_type.startswith('destructive_dispenser'):
        if previous_agent_item == 'tomato':
            return 14, actionMap[14]  # 'put down tomato on dispenser'
        elif previous_agent_item == 'plate':
            return 15, actionMap[15]  # 'put down plate on dispenser'
        elif previous_agent_item == 'tomato_cut':
            return 17, actionMap[17]  # 'put down tomato_cut on dispenser'
        elif previous_agent_item == 'tomato_salad':
            return 18, actionMap[18]  # 'put down tomato_salad on dispenser'
    elif action_type in ['use_cutting_board']:
        if item_change_type == "drop":
            return 19, actionMap[19]  # 'put down tomato on cuttingboard'
        elif item_change_type == "cutting":
            return 3, actionMap[3]  # 'start cutting tomato'
        elif item_change_type == "pickup":
            return 4, actionMap[4]  # 'pick up tomato_cut from cuttingboard'
    elif action_type in ['use_delivery']:
        return 9, actionMap[9]  # 'deliver tomato_salad'
    elif action_type.startswith('put_down_item_on_free_counter'):
        if previous_agent_item == 'plate':
            return 0, actionMap[0]  # 'put down plate on counter'
        elif previous_agent_item == 'tomato':
            return 6, actionMap[6]  # 'put down tomato on counter'
        elif previous_agent_item == 'tomato_cut':
            return 11, actionMap[11]  # 'put down tomato_cut on counter'
        elif previous_agent_item == 'tomato_salad':
            return 13, actionMap[13]  # 'put down tomato_salad on counter'
    elif action_type in ['pick_up_plate_from_counter_closest', 'pick_up_plate_from_counter_midpoint']:
        return 10, actionMap[10]  # 'pick up plate from counter'
    elif action_type in ['pick_up_tomato_from_counter_closest', 'pick_up_tomato_from_counter_midpoint']:
        return 8, actionMap[8]  # 'pick up tomato from counter'
    elif action_type in ['pick_up_tomato_cut_from_counter_closest', 'pick_up_tomato_cut_from_counter_midpoint']:
        return 12, actionMap[12]  # 'pick up tomato_cut from counter'
    elif action_type in ['pick_up_tomato_salad_from_counter_closest', 'pick_up_tomato_salad_from_counter_midpoint']:
        return 5, actionMap[5]  # 'pick up tomato_salad from counter'

    # Special case for assembly
    elif item_change_type == "assembly":
        return 7, actionMap[7]  # 'assemble salad'

    # Fallback
    return -1, f"UNKNOWN: {item_change_type} {previous_agent_item} -> {current_agent_item} at {target_tile_type}"

def analyze_meaningful_actions(actions_df, simulation_df, counter_df, map_nr, output_dir=None, engine_tick_rate=24, cutting_speeds=None):
    """Detect meaningful actions by finding item state changes and matching to actions.csv"""
    
    # Read the CSV files if they are paths, otherwise use the dataframes directly
    if isinstance(actions_df, (str, Path)):
        print("Reading CSV files...")
        actions_df = pd.read_csv(actions_df)
        simulation_df = pd.read_csv(simulation_df)
        counter_df = pd.read_csv(counter_df)
    
    print(f"Actions shape: {actions_df.shape}")
    print(f"Simulation shape: {simulation_df.shape}")
    print(f"Counter shape: {counter_df.shape}")
    
    # DEBUG: Check for NaN values in the data
    print("\n=== DEBUGGING NaN VALUES ===")
    
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
    
    # Remove NaN values from actions DataFrame for target_tile_x and target_tile_y
    actions_df = actions_df[pd.notna(actions_df['target_tile_x']) & pd.notna(actions_df['target_tile_y'])].reset_index(drop=True)

    print("=== END NaN DEBUGGING ===\n")
        
    # Sort by agent_id and action_id (chronological order)
    actions_df = actions_df.sort_values(['agent_id', 'action_id']).reset_index(drop=True)
    
    # Sort simulation and counter by frame
    simulation_df = simulation_df.sort_values('frame').reset_index(drop=True)
    counter_df = counter_df.sort_values('frame').reset_index(drop=True)

    available_counters = {}
    for i, col in enumerate(counter_df.columns[2:], start=2):  # position starts at 2
        # column name format: counter_x_y
        parts = col.split("_")
        if len(parts) == 3 and parts[0] == "counter":
            x, y = int(parts[1]), int(parts[2])
            available_counters[i] = (x, y)

    meaningful_actions = []
    
    # Track states for each agent
    agent_states = {}
    for agent_id in actions_df['agent_id'].unique():
        agent_states[agent_id] = {
            'current_action_idx': 0,
            'previous_agent_item': None,
            'actions': actions_df[actions_df['agent_id'] == agent_id].copy().reset_index(drop=True),
            'skip_frames': set()  # Frames to skip processing (already handled in compound actions)
        }
        print(f"Agent {agent_id} has {len(agent_states[agent_id]['actions'])} actions")
    
    # Process all agents simultaneously, frame by frame
    for frame in sorted(simulation_df['frame'].unique()):
        frame_data = simulation_df[simulation_df['frame'] == frame]
        counter_states_prev = counter_df[counter_df['frame'] == frame - 1] if frame > 0 else None
        counter_states = counter_df[counter_df['frame'] == frame]

        print(f"\n=== Processing Frame {frame}, Second {frame / engine_tick_rate} ===")        
        
        # Process each agent in this frame
        for _, sim_row in frame_data.iterrows():
            agent_id = sim_row['agent_id']
            current_agent_item = sim_row['item'] if pd.notna(sim_row['item']) else None
            previous_agent_item = agent_states[agent_id]['previous_agent_item']

            # Skip frames that are already handled (e.g., pickup part of cuttingboard sequence)
            if frame in agent_states[agent_id]['skip_frames']:
                print(f"  Agent {agent_id}: Skipping frame {frame} (already processed as part of previous action)")
                # Still update the previous_agent_item even when skipping to maintain state consistency
                agent_states[agent_id]['previous_agent_item'] = current_agent_item
                continue
            
            # DEBUG: Check this specific row for NaN values
            if pd.isna(sim_row['tile_x']) or pd.isna(sim_row['tile_y']):
                print(f"      WARNING: Agent {agent_id} has NaN coordinates at frame {frame}, skipping")
                continue
            else:
                agent_tile_x = int(sim_row['tile_x'])
                agent_tile_y = int(sim_row['tile_y'])
            
            # Detect item state change
            item_changed = False
            change_type = ""
            
            if previous_agent_item != current_agent_item:
                if previous_agent_item is None and current_agent_item is not None:
                    item_changed = True
                    change_type = "pickup"
                elif previous_agent_item is not None and current_agent_item is None:
                    item_changed = True
                    change_type = "drop"
                elif previous_agent_item != current_agent_item:
                    item_changed = True
                    change_type = "change"

            print(f"  Agent {agent_id}: {previous_agent_item} -> {current_agent_item} (change_type: {change_type if item_changed else 'no_change'}) at tile ({agent_tile_x}, {agent_tile_y})")

            if item_changed:
                action_found = False
                print(f"      Item changed detected: {change_type}")
                
                # Find if agent is near any counter
                nearby_counters = find_nearby_tiles(agent_tile_x, agent_tile_y, 
                                                   agent_states[agent_id]['actions'], ['counter'])

                # Track transitions for each nearby counter
                counter_transitions = {}

                # Check for counter use if there has been a change in item on nearby counters
                if counter_states_prev is not None and nearby_counters:
                    for counter in nearby_counters:
                        counter_pos = (int(counter['x']), int(counter['y']))
                        
                        counter_col_name = f'counter_{counter_pos[0]}_{counter_pos[1]}'
                        if counter_col_name in counter_states_prev.columns and counter_col_name in counter_states.columns:
                            prev_counter_item = counter_states_prev[counter_col_name].iloc[0]
                            curr_counter_item = counter_states[counter_col_name].iloc[0]
                            prev_counter_item = None if pd.isna(prev_counter_item) else prev_counter_item
                            curr_counter_item = None if pd.isna(curr_counter_item) else curr_counter_item
                        else:
                            continue  # Skip if counter column doesn't exist
                            
                        if prev_counter_item != curr_counter_item:
                            transition_type = None
                            search_action = None
                            
                            if curr_counter_item == 'tomato_salad' and previous_agent_item != 'tomato_salad':
                                if previous_agent_item == 'plate' and prev_counter_item == 'tomato_cut':
                                    search_action = 'pick_up_tomato_cut_from_counter'
                                elif previous_agent_item == 'tomato_cut' and prev_counter_item == 'plate':
                                    search_action = 'pick_up_plate_from_counter'
                                # action is assembling salad on counter
                                transition_type = 'salad_assembly'
                                print(f"      Detected salad assembly at counter {counter_pos}")
                            elif curr_counter_item == previous_agent_item and prev_counter_item == current_agent_item and curr_counter_item is not None and prev_counter_item is not None:
                                # action is swapping items on counter (combined action tracking)
                                transition_type = 'swap_at_counter'
                                print(f"      Detected item swap at counter {counter_pos}")
                            elif curr_counter_item == previous_agent_item and prev_counter_item is None and curr_counter_item is not None:
                                # action is putting down item on counter
                                transition_type = 'drop_at_counter'
                                print(f"      Detected drop at counter {counter_pos}")
                            elif prev_counter_item == current_agent_item and curr_counter_item is None and prev_counter_item is not None:
                                # action is picking up item from counter
                                transition_type = 'pick_up_from_counter'
                                print(f"      Detected pick up from counter {counter_pos}")
                                                        
                            if transition_type:
                                counter_transitions[counter_pos] = {
                                    'type': transition_type,
                                    'search_action': search_action,
                                    'prev_item': prev_counter_item,
                                    'curr_item': curr_counter_item
                                }

                # Get current action index and actions for this agent
                current_action_idx = agent_states[agent_id]['current_action_idx']
                agent_actions = agent_states[agent_id]['actions']
                
                # Process counter transitions
                for counter_pos, transition_info in counter_transitions.items():
                    transition_type = transition_info['type']
                    search_action = transition_info['search_action']
                    
                    if transition_type == 'salad_assembly' and not action_found:
                        print(f"      Processing salad assembly action at counter {counter_pos}")
                        # Find the action that matches this position and is related to counter operations
                        matched_action = None
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (actions without specific positions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                                
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])
                            
                            # Look for any counter-related action (put_down or pick_up)
                            if (is_near and action['action_type'].startswith(search_action) 
                                and action['target_tile_x'] == counter_pos[0] and action['target_tile_y'] == counter_pos[1]
                                ):
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                                break
                        
                        if matched_action is not None:                        
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                                                    
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
                                'item_change_type': "drop",
                                'previous_item': previous_agent_item,
                                'current_item': None,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': 7,
                                'action_category_name': "assemble salad",
                                'compound_action_part': 0  # Single assembly action
                            }
                            meaningful_actions.append(assembly_action)

                            print(f"      Recorded salad assembly action: assemble salad at counter {counter_pos}")
                            action_found = True
                            break  # Process only one transition per frame
                        else:
                            print(f"      Could not match salad assembly action at counter {counter_pos}")
                    
                    elif transition_type == 'swap_at_counter' and not action_found:
                        print(f"      Processing swap at counter {counter_pos}")

                        # Find the action that matches this position and is related to counter operations
                        matched_action = None
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (actions without specific positions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                                
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])
                            
                            # Look for any counter-related action (put_down or pick_up)
                            if (is_near and action['action_type'].startswith(f'pick_up_{current_agent_item}_from_counter') 
                                and action['target_tile_x'] == counter_pos[0] and action['target_tile_y'] == counter_pos[1]
                                ):
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                                break
                        
                        if matched_action is not None:                       
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                                        
                            # 1. Put down the previous item (automatic action)
                            put_down_category_id, put_down_category_name = get_action_category(
                                'put_down_item_on_free_counter', "drop", previous_agent_item, None, matched_action['target_tile_type']
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
                                'previous_item': previous_agent_item,
                                'current_item': None,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': put_down_category_id,
                                'action_category_name': put_down_category_name,
                                'compound_action_part': 1  # First part of compound action
                            }
                            meaningful_actions.append(put_down_action)
                            print(f"      Part 1 - Put down: {put_down_category_name}")

                            # 2. Pick up the new item
                            pick_up_category_id, pick_up_category_name = get_action_category(
                                matched_action['action_type'], "pickup", None, current_agent_item, matched_action['target_tile_type']
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
                                'current_item': current_agent_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': pick_up_category_id,
                                'action_category_name': pick_up_category_name,
                                'compound_action_part': 2  # Second part of compound action
                            }
                            meaningful_actions.append(pick_up_action)
                            print(f"      Part 2 - Pick up: {pick_up_category_name}")
                            action_found = True
                            break  # Process only one transition per frame
                        else:
                            print(f"      Could not match swap on counter action at counter {counter_pos}")

                    elif transition_type == 'drop_at_counter' and not action_found:
                        print(f"      Processing drop at counter {counter_pos}")

                        # Find the action that matches this position and is related to counter operations
                        matched_action = None
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (actions without specific positions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                                
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])
                            
                            if (is_near and action['action_type'].startswith(f'put_down_item_on_free_counter') 
                                and action['target_tile_x'] == counter_pos[0] and action['target_tile_y'] == counter_pos[1]
                                ):
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                                break
                        
                        if matched_action is not None:                       
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1

                            category_id, category_name = get_action_category(
                                matched_action['action_type'], "drop", previous_agent_item, None, matched_action['target_tile_type']
                            )

                            action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'item_change_type': "drop",
                                'previous_item': previous_agent_item,
                                'current_item': None,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': category_id,
                                'action_category_name': category_name,
                                'compound_action_part': 0  # Single action
                            }
                            meaningful_actions.append(action)
                            print(f"      Drop: {category_name}")
                            action_found = True
                            break  # Process only one transition per frame
                        
                        else:
                            print(f"      Could not match drop on counter action at counter {counter_pos}")

                    elif transition_type == 'pick_up_from_counter' and not action_found:
                        print(f"      Processing pick up from counter {counter_pos}")

                        # Find the action that matches this position and is related to counter operations
                        matched_action = None
                        for check_idx in range(current_action_idx, len(agent_actions)):
                            action = agent_actions.iloc[check_idx]
                            
                            # Skip actions with NaN coordinates (actions without specific positions)
                            if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                                continue
                                
                            is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                   action['target_tile_x'], action['target_tile_y'])

                            if (is_near and action['action_type'].startswith(f'pick_up_{current_agent_item}_from_counter')
                                and action['target_tile_x'] == counter_pos[0] and action['target_tile_y'] == counter_pos[1]
                                ):
                                matched_action = action.copy()
                                matched_action['matched_action_idx'] = check_idx
                                print(f"      Matched to action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                                break
                        
                        if matched_action is not None:                       
                            agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1

                            category_id, category_name = get_action_category(
                                matched_action['action_type'], "pickup", previous_agent_item, None, matched_action['target_tile_type']
                            )

                            action = {
                                'frame': sim_row['frame'],
                                'agent_id': agent_id,
                                'action_id': matched_action['action_id'],
                                'action_number': matched_action['action_number'],
                                'action_type': matched_action['action_type'],
                                'target_tile_type': matched_action['target_tile_type'],
                                'target_tile_x': matched_action['target_tile_x'],
                                'target_tile_y': matched_action['target_tile_y'],
                                'agent_tile_x': agent_tile_x,
                                'item_change_type': "pickup",
                                'previous_item': None,
                                'current_item': current_agent_item,
                                'agent_x': sim_row['x'],
                                'agent_y': sim_row['y'],
                                'action_category_id': category_id,
                                'action_category_name': category_name,
                                'compound_action_part': 0  # Single action
                            }
                            meaningful_actions.append(action)
                            print(f"      Pick up: {category_name}")
                            action_found = True
                            break  # Process only one transition per frame
                        
                        else:
                            print(f"      Could not match pick up on counter action at counter {counter_pos}")

                # If no counter transitions were processed, check for other action types
                if change_type == "change" and current_agent_item in ['plate', 'tomato'] and not action_found:
                    print(f"      Checking for dispenser action (destructive): {previous_agent_item} -> {current_agent_item}")
                    print(f"      Current action index: {current_action_idx}/{len(agent_actions)}")

                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]

                        # Skip actions with NaN coordinates for now
                        if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                            continue
                                
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                action['target_tile_x'], action['target_tile_y'])

                        if is_near and (
                            (action['action_type'] == 'pick_up_plate_from_dispenser' and current_agent_item == 'plate')
                              or (action['action_type'] == 'pick_up_tomato_from_dispenser' and current_agent_item == 'tomato')
                            ):
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            print(f"      Matched to dispenser action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                            break
                            
                    if matched_action is not None:
                        print(f"     Processing as destructive action at dispenser")
                        # Update action index AFTER processing
                        agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1
                        
                        # 1. Put down the previous item (destructive action)
                        put_down_category_id, put_down_category_name = get_action_category(
                            'destructive_dispenser', "drop", previous_agent_item, None, matched_action['target_tile_type']
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
                            'previous_item': previous_agent_item,
                            'current_item': None,
                            'agent_x': sim_row['x'],
                            'agent_y': sim_row['y'],
                            'action_category_id': put_down_category_id,
                            'action_category_name': put_down_category_name,
                            'compound_action_part': 1  # First part of compound action
                        }
                        meaningful_actions.append(put_down_action)
                        print(f"      Part 1 - Put down: {put_down_category_name}")
                        
                        # 2. Pick up the new item
                        pick_up_category_id, pick_up_category_name = get_action_category(
                            matched_action['action_type'], "pickup", None, current_agent_item, matched_action['target_tile_type']
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
                            'current_item': current_agent_item,
                            'agent_x': sim_row['x'],
                            'agent_y': sim_row['y'],
                            'action_category_id': pick_up_category_id,
                            'action_category_name': pick_up_category_name,
                            'compound_action_part': 2  # Second part of compound action
                        }
                        meaningful_actions.append(pick_up_action)
                        print(f"      Part 2 - Pick up: {pick_up_category_name}")
                        action_found = True

                    else:
                        print(f"      No dispenser found - checking for counter-based transformation")
                
                # Handle cuttingboard mechanics separately
                elif change_type == "drop" and previous_agent_item == 'tomato' and current_agent_item is None and not action_found:
                    print(f"      Checking if this is cuttingboard tomato drop")
                    
                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]

                        # Skip actions with NaN coordinates for now
                        if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                            continue
                                
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                action['target_tile_x'], action['target_tile_y'])

                        if is_near and action['action_type'] == 'use_cutting_board':
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            print(f"      Matched to cutting_board action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                            break
                    
                    if matched_action is not None:                        
                        print(f"      Processing as cuttingboard usage")
                        # Update action index AFTER processing
                        agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1

                        # Step 1: Put down tomato on cuttingboard
                        put_down_category_id, put_down_category_name = get_action_category(
                            matched_action['action_type'], "drop", previous_agent_item, None, matched_action['target_tile_type']
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
                            'previous_item': previous_agent_item,
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
                        start_cutting_category_id, start_cutting_category_name = get_action_category(
                            matched_action['action_type'], "cutting", None, None, matched_action['target_tile_type']
                        )
                        
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
                            'action_category_id': start_cutting_category_id,
                            'action_category_name': start_cutting_category_name,
                            'compound_action_part': 2  # Second part of cuttingboard sequence
                        }
                        meaningful_actions.append(start_cutting_action)
                        print(f"      Cuttingboard Step 2: {start_cutting_category_name}")
                        
                        # Step 3: Pick up tomato_cut from cuttingboard (use future frame)
                        # Look for the None -> tomato_cut transition in frames between current and target
                        # Get cutting speed for this agent
                        agent_cutting_speed = cutting_speeds.get(agent_id, 1.0) if cutting_speeds else 1.0
                        cutting_time = 3.0  # Base cutting time in seconds
                        actual_cutting_time = cutting_time / agent_cutting_speed
                        target_frame = int(sim_row['frame'] + (actual_cutting_time + 2.0) * engine_tick_rate)  # Add 2 second buffer
                        search_frames = range(sim_row['frame'] + 1, target_frame)  # Look ahead up to target + small buffer

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

                                # Check if agent is still on the same tile and gets tomato_cut
                                if future_tile_x == agent_tile_x and future_tile_y == agent_tile_y:
                                    if future_item == 'tomato_cut':
                                        print(f"      Verified cuttingboard usage: Agent {agent_id} at frame {check_frame} on tile ({agent_tile_x}, {agent_tile_y}) with tomato_cut")
                                        future_frame = check_frame
                                        agent_states[agent_id]['skip_frames'].add(future_frame)
                                        break
    
                        if future_frame:
                            pick_up_category_id, pick_up_category_name = get_action_category(
                                matched_action['action_type'], "pickup", None, future_item, matched_action['target_tile_type']
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
                            
                            # Mark the future frame to skip processing the tomato_cut pickup later
                            if agent_id not in agent_states:
                                agent_states[agent_id] = {}
                            if 'skip_frames' not in agent_states[agent_id]:
                                agent_states[agent_id]['skip_frames'] = set()
                            agent_states[agent_id]['skip_frames'].add(future_frame)
                        
                        else:
                            print(f"      No None -> tomato_cut transition found for agent {agent_id} between frames {sim_row['frame']} and {target_frame}")

                        action_found = True
                    else:
                        print(f"      Could not match drop action - not a cuttingboard action")
                
                # Handle delivery mechanics
                elif change_type == "drop" and previous_agent_item == 'tomato_salad' and current_agent_item is None and not action_found:
                    print(f"      Checking if this is a delivery")
                    
                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]

                        # Skip actions with NaN coordinates for now
                        if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                            continue
                                
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                action['target_tile_x'], action['target_tile_y'])

                        if is_near and action['action_type'] == 'use_delivery':
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            print(f"      Matched to delivery action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                            break
                    
                    if matched_action is not None:                        
                        print(f"      Processing as cuttingboard usage")
                        # Update action index AFTER processing
                        agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1

                        category_id, category_name = get_action_category(
                            matched_action['action_type'], "drop", previous_agent_item, None, matched_action['target_tile_type']
                        )
                        
                        action = {
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
                            'previous_item': previous_agent_item,
                            'current_item': None,
                            'agent_x': sim_row['x'],
                            'agent_y': sim_row['y'],
                            'action_category_id': category_id,
                            'action_category_name': category_name,
                            'compound_action_part': 0 # Single action
                        }
                        meaningful_actions.append(action)
                        print(f"      Delivery: {category_name}")
                        action_found = True
                    else:
                        print(f"      Could not match drop action - not a delivery action")

                elif change_type == "pickup" and previous_agent_item is None and current_agent_item == 'tomato_cut' and not action_found:
                    # Check if this pickup should be skipped (already processed as part of cuttingboard sequence)
                    if (agent_id in agent_states and 
                        'skip_frames' in agent_states[agent_id] and 
                        sim_row['frame'] in agent_states[agent_id]['skip_frames']):
                        print(f"      Skipping tomato_cut pickup - already processed as part of cuttingboard sequence")
                        agent_states[agent_id]['skip_frames'].remove(sim_row['frame'])
                        continue
                    else:
                        print(f"      Could not process standalone tomato_cut pickup")
                        continue
                
                # Handle dispenser pickups
                elif change_type == "pickup" and previous_agent_item is None and current_agent_item in ['plate', 'tomato'] and not action_found:
                    print(f"      Checking for dispenser pickup: {current_agent_item}")
                    print(f"      Current action index: {current_action_idx}/{len(agent_actions)}")

                    matched_action = None
                    for check_idx in range(current_action_idx, len(agent_actions)):
                        action = agent_actions.iloc[check_idx]

                        # Skip actions with NaN coordinates for now
                        if pd.isna(action['target_tile_x']) or pd.isna(action['target_tile_y']):
                            continue
                                
                        is_near = is_near_target(agent_tile_x, agent_tile_y, 
                                                action['target_tile_x'], action['target_tile_y'])

                        if is_near and (
                            (action['action_type'] == 'pick_up_plate_from_dispenser' and current_agent_item == 'plate')
                              or (action['action_type'] == 'pick_up_tomato_from_dispenser' and current_agent_item == 'tomato')
                            ):
                            matched_action = action.copy()
                            matched_action['matched_action_idx'] = check_idx
                            print(f"      Matched to dispenser pickup action {check_idx}: {action['target_tile_type']} -> ({action['target_tile_x']}, {action['target_tile_y']})")
                            break
                    
                    if matched_action is not None:
                        print(f"     Processing as dispenser pickup")
                        # Update action index AFTER processing
                        agent_states[agent_id]['current_action_idx'] = matched_action['matched_action_idx'] + 1

                        category_id, category_name = get_action_category(
                            matched_action['action_type'], "pickup", None, current_agent_item, matched_action['target_tile_type']
                        )

                        action = {
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
                            'current_item': current_agent_item,
                            'agent_x': sim_row['x'],
                            'agent_y': sim_row['y'],
                            'action_category_id': category_id,
                            'action_category_name': category_name,
                            'compound_action_part': 0  # Single action
                        }
                        meaningful_actions.append(action)
                        print(f"      Dispenser Pickup: {category_name}")
                        action_found = True

                    else:
                        print(f"      Could not match dispenser pick up action")
                        continue

                elif not action_found:
                    print(f"      Could not match {change_type} action: {previous_agent_item} -> {current_agent_item}")
                        
            # Update the previous item for this agent
            agent_states[agent_id]['previous_agent_item'] = current_agent_item
    
    # Convert to DataFrame
    meaningful_df = pd.DataFrame(meaningful_actions)
    
    if not meaningful_df.empty:
        # Sort by frame to ensure chronological order
        meaningful_df = meaningful_df.sort_values(['frame', 'agent_id']).reset_index(drop=True)
        
        # Convert target tile coordinates to integers
        meaningful_df['target_tile_x'] = meaningful_df['target_tile_x'].astype(int)
        meaningful_df['target_tile_y'] = meaningful_df['target_tile_y'].astype(int)
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
def analyze_meaningful_actions_from_files(actions_csv_path, simulation_csv_path, counter_csv_path, map_nr, output_dir=None, cutting_speeds=None):
    """Analyze meaningful actions from CSV file paths (for backward compatibility)"""
    return analyze_meaningful_actions(actions_csv_path, simulation_csv_path, counter_csv_path, map_nr, output_dir, cutting_speeds=cutting_speeds)


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze meaningful actions from simulation data')
    parser.add_argument('--actions_csv', required=True, help='Path to actions.csv file')
    parser.add_argument('--simulation_csv', required=True, help='Path to simulation.csv file')
    parser.add_argument('--counter_csv', required=True, help='Path to counter.csv file')
    parser.add_argument('--map', required=True, help='Map number or name')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--engine_tick_rate', type=int, default=24, help='Engine tick rate (FPS)')
    parser.add_argument('--cutting_speeds', type=str, help='JSON string of cutting speeds for each agent (optional)')
    
    args = parser.parse_args()
    
    # Parse cutting speeds if provided
    cutting_speeds = None
    if args.cutting_speeds:
        import json
        try:
            cutting_speeds = json.loads(args.cutting_speeds)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON for cutting_speeds, using defaults")
            cutting_speeds = None
    
    # Run analysis
    actions_csv = Path(args.actions_csv)
    simulation_csv = Path(args.simulation_csv)
    counter_csv = Path(args.counter_csv)
    map_nr = args.map
    output_dir = Path(args.output_dir)
    engine_tick_rate = args.engine_tick_rate

    result = analyze_meaningful_actions(actions_csv, simulation_csv, counter_csv, map_nr, output_dir, engine_tick_rate=engine_tick_rate, cutting_speeds=cutting_speeds)
    print("\nAnalysis complete!")