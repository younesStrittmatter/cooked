"""
Actions extraction utilities for human-like analysis of simulation data.

This module provides functions to extract and format action data
in formats commonly used by human researchers for analysis.

Author: Samuel Lozano
"""

import re
import pandas as pd
import math
from pathlib import Path


def generate_agent_action_files(meaningful_actions_df, positions_dir, output_dir, map_name=None, simulation_id=None, engine_tick_rate=24, agent_initialization_period=0.0):
    """
    Generate action CSV files for each agent from meaningful_actions.csv and position data.
    
    Creates individual CSV files for each agent with comprehensive item tracking including:
    - Basic columns: second, item, item_id, action, target_type, target_position, action_long, 
      player_id, map_name, game_id, distance_walked, distance_walked_since_last_action, 
      overall_score, player_score_change, player_score, walking_speed, cutting_speed, start_pos
    - Item tracking: last_touched, touched_list, tomato_id, tomato_history, plate_id, plate_history, 
      tomato_cut_id, tomato_cut_history, tomato_salad_id, tomato_salad_history,
      is_item_collaboration, is_exchange_collaboration,
    - Collaboration tracking: who_picked_tomato, who_picked_plate, who_cutted, who_assembled, 
      who_delivered, proportion_of_collaboration
    - Counter usage: number_of_counters_used, number_of_counters_tomato, number_of_counters_plate, 
      number_of_counters_tomato_cut, number_of_counters_tomato_salad, history_of_counters_used, 
      history_of_counters_tomato, history_of_counters_plate, history_of_counters_tomato_cut, 
      history_of_counters_tomato_salad
    
    This function processes all meaningful actions without filtering and uses direct frame 
    comparison with position data to ensure accurate matching.
    
    Args:
        meaningful_actions_df: DataFrame with meaningful action data or path to meaningful_actions.csv file
        positions_dir: Directory containing {agent_id}_positions.csv files or single position DataFrame
        output_dir: Directory to save the action files
        map_name: Name of the map (optional, will extract from file if not provided)
        simulation_id: Simulation ID (optional, will use directory name if not provided)
        engine_tick_rate: Engine tick rate from simulation config (default 24)
        agent_initialization_period: Duration of agent initialization period in seconds (default 15.0)
        
    Returns:
        Dictionary with paths to generated action files {agent_id: filepath}
    """
    
    # Read the CSV files if they are paths, otherwise use the dataframes directly
    if isinstance(meaningful_actions_df, (str, Path)):
        meaningful_actions_df = pd.read_csv(meaningful_actions_df)
    
    # Load position data from individual files or use provided data
    if isinstance(positions_dir, (str, Path)):
        positions_dir = Path(positions_dir)
        position_files = {}
        # Load individual position files
        for agent_file in positions_dir.glob("*_positions.csv"):
            agent_id = agent_file.stem.replace('_positions', '')
            if agent_id.startswith('ai_rl_'):
                position_files[agent_id] = pd.read_csv(agent_file)
    else:
        # positions_dir is actually a dict of DataFrames
        position_files = positions_dir
        
    output_dir = Path(output_dir)
    action_files = {}
    
    # Check if meaningful_actions_df has the required columns
    required_columns = ['agent_id', 'frame', 'action_category_name']
    available_columns = meaningful_actions_df.columns.tolist()
    missing_columns = [col for col in required_columns if col not in available_columns]
    
    if missing_columns:
        print(f"Error: meaningful_actions_df is missing required columns: {missing_columns}")
        print(f"Available columns: {available_columns}")
        return action_files
    
    # Extract simulation_id from output directory if not provided
    if simulation_id is None:
        simulation_id = output_dir.name
    
    # Extract map_name from simulation_id if not provided
    if map_name is None:
        # Try to extract map name from simulation_id (assuming format contains map name)
        map_name = "unknown_map"
        
    # Sort meaningful actions
    meaningful_actions_df = meaningful_actions_df.sort_values(['agent_id', 'frame']).reset_index(drop=True)
    
    if meaningful_actions_df.empty:
        print(f"Warning: No meaningful actions found")
        return {}
    
    print(f"Processing all meaningful actions: {len(meaningful_actions_df)}")
    
    # Use engine tick rate from simulation config
    ENGINE_TICK_RATE = engine_tick_rate
    
    # Three registry tables for organized item tracking
    id_connection = {}      # item_id -> {tomato_id, plate_id, tomato_cut_id, tomato_salad_id}
    item_history = {}       # item_id -> {touched_list, last_touched, counter_history, counter_count}
    item_actions = {}       # item_id -> {who_picked_tomato, who_picked_plate, who_cutted, who_assembled, who_delivered}
    
    item_index = {'tomato': 1, 'plate': 1, 'tomato_cut': 1, 'tomato_salad': 1}  # Item ID counters
        
    def create_item_id(item_type):
        """Create unique item identifier and increment counter."""
        item_id = f"{item_type}_{item_index[item_type]}"
        item_index[item_type] += 1
        return item_id
    
    def register_new_item(item_id, item_type, agent_id, origin_item_id=None):
        """Register a new item in the three registry tables."""        
        # Initialize id_connection entry
        if item_type == 'tomato':
            id_connection[item_id] = {
                'tomato_id': item_id,
                'plate_id': '',
                'tomato_cut_id': '',
                'tomato_salad_id': ''
            }
        elif item_type == 'plate':
            id_connection[item_id] = {
                'tomato_id': '',
                'plate_id': item_id,
                'tomato_cut_id': '',
                'tomato_salad_id': ''
            }
        elif item_type == 'tomato_cut':
            # Get tomato_id from origin
            tomato_id = origin_item_id if origin_item_id else ''
            if tomato_id and tomato_id in id_connection:
                # Copy tomato info and add tomato_cut
                id_connection[item_id] = id_connection[tomato_id].copy()
                id_connection[item_id]['tomato_cut_id'] = item_id
            else:
                id_connection[item_id] = {
                    'tomato_id': tomato_id,
                    'plate_id': '',
                    'tomato_cut_id': item_id,
                    'tomato_salad_id': ''
                }
        elif item_type == 'tomato_salad':
            # origin_item_id contains the complete origin information
            if origin_item_id and isinstance(origin_item_id, dict):
                id_connection[item_id] = {
                    'tomato_id': origin_item_id.get('tomato_id', ''),
                    'plate_id': origin_item_id.get('plate_id', ''),
                    'tomato_cut_id': origin_item_id.get('tomato_cut_id', ''),
                    'tomato_salad_id': item_id
                }
            else:
                id_connection[item_id] = {
                    'tomato_id': '',
                    'plate_id': '',
                    'tomato_cut_id': '',
                    'tomato_salad_id': item_id
                }
        
        # Initialize item_history entry
        item_history[item_id] = {
            'touched_list': [agent_id],
            'last_touched': agent_id,
            'counter_history': [],
            'counter_count': 0
        }
        
        # Initialize item_actions entry
        if item_type in ['tomato', 'plate']:
            item_actions[item_id] = {f'who_picked_{item_type}': agent_id}
        elif item_type == 'tomato_cut':
            item_actions[item_id] = {'who_cutted': agent_id}
        elif item_type == 'tomato_salad':
            item_actions[item_id] = {'who_assembled': agent_id, 'who_delivered': ''}
        
        return item_id

    def update_item_interaction(item_id, agent_id, second, action_type, target_type=None, target_location=None):
        """Update item interaction tracking in the item_history table."""
        if item_id not in item_history:
            return None
            
        history_data = item_history[item_id]
        
        # Update touched list - ensure agent is recorded when they interact with item
        if agent_id not in history_data['touched_list'] or agent_id != history_data['last_touched']:
            if agent_id not in history_data['touched_list']:
                history_data['touched_list'].append(agent_id)
            history_data['last_touched'] = agent_id
        
        # Track counter interactions (place/pickup)
        if action_type in ['drop', 'pickup'] and target_type == 'counter':
            history_data['counter_history'].append((agent_id, action_type, second, target_location))
            if action_type == 'drop':
                history_data['counter_count'] += 1
        
        return history_data
    
    # Global tracking of items by location
    counter_items = {}  # (x, y) -> {item_id, type, second_dropped}

    def create_salad(item_involved, current_item_id, agent_id, second, target_location):
        # When assembling salad, agent drops tomato_cut or plate on counter
        if target_location is None:
            print(f"Warning: Cannot create salad without target location")
            return
            
        origin_ids = {}
        if item_involved == 'tomato_cut':
            origin_ids['tomato_cut_id'] = current_item_id
            # Get the tomato_id associated with this tomato_cut_id from the connections table
            tomato_cut_connection = id_connection.get(current_item_id, {})
            origin_ids['tomato_id'] = tomato_cut_connection.get('tomato_id', '')
            if target_location in counter_items:
                origin_ids['plate_id'] = counter_items[target_location]['item_id']
        elif item_involved == 'plate':
            origin_ids['plate_id'] = current_item_id
            if target_location in counter_items:
                origin_ids['tomato_cut_id'] = counter_items[target_location]['item_id']
                # Get the tomato_id associated with this tomato_cut_id from the connections table
                tomato_cut_connection = id_connection.get(counter_items[target_location]['item_id'], {})
                origin_ids['tomato_id'] = tomato_cut_connection.get('tomato_id', '')

        # Create the salad on the counter but don't return it as current item
        salad_id = create_item_id('tomato_salad')
        register_new_item(salad_id, 'tomato_salad', agent_id, origin_ids)
        counter_items[target_location] = {'item_id': salad_id, 'type': 'tomato_salad', 'second_dropped': second}

    def get_pick_up_item_id(current_item_name, agent_id, second, target_type, target_location=None):
        """Determine the current item ID based on action and item names. Track items by location."""
        item_id = None  # Initialize item_id to ensure it's always defined
        
        if target_type == 'cuttingboard' and current_item_name == 'tomato_cut':
            item_id = create_item_id('tomato_cut')
            origin_tomato_id = cutting_registry.get(agent_id)
            register_new_item(item_id, 'tomato_cut', agent_id, origin_tomato_id)
            if agent_id in cutting_registry:
                del cutting_registry[agent_id]
            else:
                print(f"Warning: Unable to handle {current_item_name} pickup from {target_type} at {second}s")

        else:
            # Create new tomato/plate when picked from dispenser
            if target_type == 'dispenser' and current_item_name in ['tomato', 'plate']:
                item_id = create_item_id(current_item_name)
                register_new_item(item_id, current_item_name, agent_id, None)
            elif target_type == 'counter':
                if target_location is not None and target_location in counter_items and counter_items[target_location]['type'] == current_item_name:
                    item_id = counter_items[target_location]['item_id']
                elif compound_item.get(agent_id) is not None:
                    item_id = compound_item.get(agent_id)
                    compound_item[agent_id] = None
                else:
                    # Item not found in counter tracking - check if there might be any item at this location
                    print(f"Warning: Item {current_item_name} not found at counter location {target_location} at {second}s.")
                    print(f"  Available items at this location: {counter_items.get(target_location, 'None') if target_location is not None else 'None'}")
                    print(f"  All counter items: {counter_items}")
                    item_id = None
            else:
                print(f"Warning: Unable to determine {current_item_name} item_id for target_type '{target_type}' at {second}s")
                item_id = None

        return item_id
    
    def get_item_collaboration_data(item_id, item_involved, available_agents=None):
        """Get collaboration data for a specific item using the new registry tables."""
        if item_id not in item_history:
            return {}, False, False
        
        # Default to standard agents if not provided
        if available_agents is None:
            available_agents = ['ai_rl_1', 'ai_rl_2']
        else:
            # Filter to only RL agents
            available_agents = [agent for agent in available_agents if agent.startswith('ai_rl_')]
            
        history_data = item_history[item_id]
        connection_data = id_connection.get(item_id, {})
        
        # Check if multiple agents touched this specific item
        is_item_collaboration = len(set(history_data.get('touched_list', []))) > 1
        
        # Check if multiple agents were involved in the full history
        all_involved_agents = set(history_data.get('touched_list', []))
        
        # Add agents from origin items using id_connection
        for origin_type in ['tomato_id', 'plate_id', 'tomato_cut_id']:
            origin_id = connection_data.get(origin_type, '')
            if origin_id and origin_id in item_history:
                all_involved_agents.update(item_history[origin_id].get('touched_list', []))
        
        # Check if multiple agents were involved across the entire item lineage (exchange collaboration)
        is_exchange_collaboration = len(all_involved_agents) > 1
                
        # Calculate collaboration percentages
        collaboration_data = {
            'who_picked_tomato': '',
            'who_picked_plate': '',
            'who_cutted': '',
            'who_assembled': '',
            'who_delivered': '',
            'proportion_of_collaboration': [0.0, 0.0]
        }
        
        # Get role assignments based on item history using the new registry tables
        if item_involved == 'tomato_salad':
            # For tomato_salad: get all role information
            salad_id = item_id
            tomato_cut_id = connection_data.get('tomato_cut_id', '')
            tomato_id = connection_data.get('tomato_id', '')
            plate_id = connection_data.get('plate_id', '')
            
            # Get action data for each item
            collaboration_data['who_assembled'] = item_actions.get(salad_id, {}).get('who_assembled', '')
            collaboration_data['who_delivered'] = item_actions.get(salad_id, {}).get('who_delivered', '')
            collaboration_data['who_cutted'] = item_actions.get(tomato_cut_id, {}).get('who_cutted', '')
            collaboration_data['who_picked_tomato'] = item_actions.get(tomato_id, {}).get('who_picked_tomato', '')
            collaboration_data['who_picked_plate'] = item_actions.get(plate_id, {}).get('who_picked_plate', '')

            num_roles = 5
        
        elif item_involved == 'tomato_cut':
            # For tomato_cut: get cut and tomato pick information
            tomato_cut_id = item_id
            tomato_id = connection_data.get('tomato_id', '')
            
            collaboration_data['who_cutted'] = item_actions.get(tomato_cut_id, {}).get('who_cutted', '')
            collaboration_data['who_picked_tomato'] = item_actions.get(tomato_id, {}).get('who_picked_tomato', '')
            
            num_roles = 2            

        else:
            # For basic items (tomato, plate)
            if item_id.startswith('tomato_'):
                collaboration_data['who_picked_tomato'] = item_actions.get(item_id, {}).get('who_picked_tomato', '')
            elif item_id.startswith('plate_'):
                collaboration_data['who_picked_plate'] = item_actions.get(item_id, {}).get('who_picked_plate', '')

            num_roles = 1

        agent_contributions = {}
        role_weight = 1.0 / num_roles
        for role, agent in collaboration_data.items():
            if agent and role != 'proportion_of_collaboration':
                    agent_contributions[agent] = agent_contributions.get(agent, 0) + role_weight
        
        # Use available agents instead of hardcoded list, ensure we have at least 2 entries for consistency
        agents_in_order = available_agents[:2] if len(available_agents) >= 2 else available_agents + [''] * (2 - len(available_agents))
        collaboration_data['proportion_of_collaboration'] = [
            agent_contributions.get(agents_in_order[0], 0.0),
            agent_contributions.get(agents_in_order[1], 0.0)
        ]
        return collaboration_data, is_item_collaboration, is_exchange_collaboration

    # Get histories for each item type
    def get_item_history(item_id):
        if item_id and item_id in item_history:
            return ';'.join(item_history[item_id].get('touched_list', []))
        return ''
        
    # Get counter usage data for the current item
    def get_item_counter_data(item_id):
        if item_id and item_id in item_history:
            counter_history = item_history[item_id].get('counter_history', [])
            counter_count = item_history[item_id].get('counter_count', 0)
            # Convert frames to seconds and create history string
            history_str = ';'.join([f"{agent}:{action}@{second:.2f}s" for agent, action, second, target_location in counter_history])
            return counter_count, history_str
        return 0, ''
    
    # Get all counter history entries sorted by time
    def get_all_counter_history_sorted(item_ids):
        all_entries = []
        for item_id in item_ids:
            if item_id and item_id in item_history:
                counter_history = item_history[item_id].get('counter_history', [])
                for agent, action, second, target_location in counter_history:
                    all_entries.append({
                        'second': second,
                        'agent': agent,
                        'action': action,
                        'item_id': item_id
                    })
        
        # Sort by second (time)
        all_entries.sort(key=lambda x: x['second'])
        
        # Create formatted history string
        history_str = ';'.join([f"{entry['agent']}:{entry['action']}@{entry['second']:.2f}s" for entry in all_entries])
        return history_str
        
    # Initialize agent-specific data structures for CSV writing
    agent_data = {}  # agent_id -> list of action records
    agent_tracking = {}  # agent_id -> tracking variables (last distance, score, etc.)
    cutting_registry = {}  # agent_id -> tomato_id (currently being cut)
    
    # Initialize tracking for each agent
    for agent_id in meaningful_actions_df['agent_id'].unique():
        if not agent_id.startswith('ai_rl_'):
            continue
            
        # Get position data for this agent
        agent_position_data = position_files.get(agent_id, pd.DataFrame())
        if agent_position_data.empty:
            print(f"Warning: No position data found for {agent_id}")
            continue
        
        # Get start position from position data 
        start_pos = agent_position_data.iloc[0]['start_pos'] if not agent_position_data.empty else "(0, 0)"
        
        # Calculate max distance for this agent
        max_distance = agent_position_data['distance_walked'].max() if not agent_position_data.empty else 0.0
        
        agent_data[agent_id] = []
        agent_tracking[agent_id] = {
            'previous_distance': 0.0,
            'previous_player_score': 0,
            'last_touched_item': 'None',
            'touched_items': [],
            'start_pos': start_pos,
            'max_distance': max_distance,
            'position_data': agent_position_data
        }
    
    # Sort all meaningful actions across all agents by frame to process chronologically
    all_actions = meaningful_actions_df.sort_values(['frame', 'agent_id']).reset_index(drop=True)
    last_item = {}  # agent_id -> {item_id}
    compound_item = {}  # agent_id -> {item_id}

    # Get available agents for collaboration data
    available_agents = [agent_id for agent_id in meaningful_actions_df['agent_id'].unique() if agent_id.startswith('ai_rl_')]

    # Process all actions chronologically
    for _, action in all_actions.iterrows():
        agent_id = action.get('agent_id')
        frame = action.get('frame', 0)
        action_category_name = action.get('action_category_name', '')
        current_item = action.get('current_item', None) if pd.notna(action.get('current_item', None)) else None
        previous_item = action.get('previous_item', None) if pd.notna(action.get('previous_item', None)) else None
        action_type = action.get('item_change_type', '')
        target_type = action.get('target_tile_type', '')
        target_x = action.get('target_tile_x', None)
        target_y = action.get('target_tile_y', None)
        compound_action_part = action.get('compound_action_part', '')
        
        # Handle None values for target coordinates
        if target_x is not None and target_y is not None:
            target_location = (int(target_x), int(target_y))  # Use tuple format for counter_items keys
            target_location_str = f'({int(target_x)}, {int(target_y)})'  # String format for CSV output
        else:
            target_location = None
            target_location_str = 'None'
        
        # Skip non-RL agents
        if not agent_id.startswith('ai_rl_') or agent_id not in agent_tracking:
            print(f"Warning: Skipping action from agent {agent_id} (frame {frame}): {action_category_name}")
            continue
        
        # Calculate timing from frame and ENGINE_TICK_RATE
        second = frame / ENGINE_TICK_RATE
        
        # Initialize agent's item tracking if not exists
        if agent_id not in last_item:
            last_item[agent_id] = None
        if agent_id not in compound_item:
            compound_item[agent_id] = None
        if agent_id not in cutting_registry:
            cutting_registry[agent_id] = None

        # Get or create item ID based on action type
        item_involved = current_item if current_item else previous_item
        existing_item_id = last_item[agent_id]
        current_item_id = None  # Initialize to ensure it's always defined
        
        # Determine current item ID based on action type
        if action_type == 'pickup':
            current_item_id = get_pick_up_item_id(item_involved, agent_id, second, target_type, target_location)
            last_item[agent_id] = current_item_id
            
            # Only remove item from counter if it's not a compound action
            # For compound actions, the item being picked up is different from what was just placed
            if target_location is not None and target_location in counter_items and compound_action_part != 2:
                del counter_items[target_location]

        elif action_type == 'cutting':
            # Find the tomato that was just placed on the cutting board and register it
            item_involved = 'tomato'
            current_item_id = cutting_registry.get(agent_id)

        elif action_type == 'drop':
            current_item_id = existing_item_id
            if target_type == 'cuttingboard' and item_involved == 'tomato':
                # Save the tomato ID for the upcoming cutting action
                cutting_registry[agent_id] = current_item_id
            elif target_type == 'delivery' and item_involved == 'tomato_salad':
                # Track delivery actions for salads
                item_actions[current_item_id]['who_delivered'] = agent_id
            elif target_type == 'counter':
                if action_category_name == 'assemble salad':
                    create_salad(item_involved, current_item_id, agent_id, second, target_location)
                else:
                    if compound_action_part == 1:
                        # Check if there's an item at the target location for compound actions
                        if target_location in counter_items:
                            compound_item[agent_id] = counter_items[target_location]['item_id']
                        else:
                            print(f"Warning: No item found at counter location {target_location} for compound action at {second}s")
                            compound_item[agent_id] = None
                    counter_items[target_location] = {'item_id': current_item_id, 'type': item_involved, 'second_dropped': second}
            last_item[agent_id] = None  # Agent's hand becomes empty after dropping

        else: 
            print(f'Warning: Unhandled action type {action_type} for item {item_involved} by agent {agent_id} at frame {frame}')
            current_item_id = existing_item_id  # Fallback to existing item

        # Update item tracking if we have a valid item ID
        update_item_interaction(current_item_id, agent_id, second, action_type, target_type, target_location)

        # Get agent tracking data
        tracking = agent_tracking[agent_id]
        if current_item_id != tracking['last_touched_item']:
            tracking['last_touched_item'] = current_item_id
            if current_item_id not in tracking['touched_items']:
                tracking['touched_items'].append(current_item_id)

        # Find corresponding position data to get score and distance
        position_data = tracking['position_data']
        position_frame_data = position_data[position_data['frame'] == frame] if 'frame' in position_data.columns else position_data[position_data['second'] <= second]
        
        if not position_frame_data.empty:
            pos_row = position_frame_data.iloc[-1]  # Get closest/latest data point
            current_player_score = pos_row.get('score', 0)
            distance_walked = pos_row.get('distance_walked', 0.0)
            
            # Calculate overall score as sum of all agents' scores at this time
            overall_score = 0
            for other_agent_id, other_tracking in agent_tracking.items():
                other_pos_data = other_tracking['position_data']
                other_frame_data = other_pos_data[other_pos_data['frame'] == frame] if 'frame' in other_pos_data.columns else other_pos_data[other_pos_data['second'] <= second]
                if not other_frame_data.empty:
                    overall_score += other_frame_data.iloc[-1].get('score', 0)
            
            if overall_score == 0:  # Fallback to current player score
                overall_score = current_player_score
        else:
            print(f"Warning: No position data found for agent {agent_id} at frame {frame}")
            # Initialize missing variables with default values
            current_player_score = 0
            overall_score = 0
            # Use approximation for distance
            total_frames = len(tracking['position_data']) if not tracking['position_data'].empty else 1
            distance_walked = (frame / (total_frames * engine_tick_rate)) * tracking['max_distance']
        
        # Calculate distance walked since last action
        distance_walked_since_last_action = distance_walked - tracking['previous_distance']
        tracking['previous_distance'] = distance_walked
        
        # Calculate player score change (difference from previous frame)
        player_score_change = current_player_score - tracking['previous_player_score']
        tracking['previous_player_score'] = current_player_score
        
        # Get item-specific data if we have a current item ID
        collaboration_data = {}
        is_item_collaboration = False
        is_exchange_collaboration = False
        
        if current_item_id and current_item_id in item_history:
            collaboration_data, is_item_collaboration, is_exchange_collaboration = get_item_collaboration_data(current_item_id, item_involved, available_agents)
        
        # Extract item IDs and histories based on current item
        last_touched = item_history.get(current_item_id, {}).get('last_touched', '') if current_item_id else ''
        touched_list_str = ';'.join(item_history.get(current_item_id, {}).get('touched_list', [])) if current_item_id else ''
        
        # Initialize item IDs and histories based on current action context
        tomato_id = ''
        tomato_history = ''
        plate_id = ''
        plate_history = ''
        tomato_cut_id = ''
        tomato_cut_history = ''
        tomato_salad_id = ''
        tomato_salad_history = ''
        
        # Determine item IDs and histories based on the current item involved in this action
        connection_data = id_connection.get(current_item_id, {}) if current_item_id else {}
        
        if item_involved == 'tomato_salad':
            # For tomato_salad: populate all related items
            tomato_salad_id = current_item_id
            tomato_salad_history = get_item_history(current_item_id)
            
            # Get related items from connection data
            tomato_id = connection_data.get('tomato_id', '')
            tomato_history = get_item_history(tomato_id)
            plate_id = connection_data.get('plate_id', '')
            plate_history = get_item_history(plate_id)
            tomato_cut_id = connection_data.get('tomato_cut_id', '')
            tomato_cut_history = get_item_history(tomato_cut_id)

        elif item_involved == 'tomato_cut':
            # For tomato_cut: populate tomato_cut and its origin tomato
            tomato_cut_id = current_item_id
            tomato_cut_history = get_item_history(current_item_id)
            
            # Get related tomato from connection data
            tomato_id = connection_data.get('tomato_id', '')
            tomato_history = get_item_history(tomato_id)
                
        elif item_involved == 'tomato':
            # For tomato: only populate tomato info
            tomato_id = current_item_id
            tomato_history = get_item_history(current_item_id)
            
        elif item_involved == 'plate':
            # For plate: only populate plate info
            plate_id = current_item_id
            plate_history = get_item_history(current_item_id)
        
        
        # Get counter data for all relevant items
        tomato_counter_count, tomato_counter_history = get_item_counter_data(tomato_id)
        plate_counter_count, plate_counter_history = get_item_counter_data(plate_id)
        tomato_cut_counter_count, tomato_cut_counter_history = get_item_counter_data(tomato_cut_id)
        tomato_salad_counter_count, tomato_salad_counter_history = get_item_counter_data(tomato_salad_id)
        
        total_counters_used = tomato_counter_count + plate_counter_count + tomato_cut_counter_count + tomato_salad_counter_count
        
        # Get all counter history sorted by frame/time for all relevant items
        relevant_item_ids = [tomato_id, plate_id, tomato_cut_id, tomato_salad_id]
        all_counter_history = get_all_counter_history_sorted(relevant_item_ids)
        
        action_data = {
            # Basic action data
            'second': second,
            'item': item_involved,
            'item_id': current_item_id,
            'action': action_type,
            'target_type': target_type,
            'target_position': target_location_str,
            'action_long': action_category_name,
            'player_id': agent_id,
            'map_name': map_name,
            'game_id': simulation_id,
            'distance_walked': distance_walked,
            'distance_walked_since_last_action': distance_walked_since_last_action,
            'overall_score': overall_score,
            'player_score_change': player_score_change,
            'player_score': current_player_score,
            'walking_speed': 1.0,
            'cutting_speed': 1.0,
            'start_pos': tracking['start_pos'],
            
            # New item tracking columns
            'last_touched': last_touched,
            'touched_list': touched_list_str,
            'tomato_id': tomato_id,
            'tomato_history': tomato_history,
            'plate_id': plate_id,
            'plate_history': plate_history,
            'tomato_cut_id': tomato_cut_id,
            'tomato_cut_history': tomato_cut_history,
            'tomato_salad_id': tomato_salad_id,
            'tomato_salad_history': tomato_salad_history,
            'is_item_collaboration': is_item_collaboration,
            'is_exchange_collaboration': is_exchange_collaboration,
            
            # Individual role tracking
            'who_picked_tomato': collaboration_data.get('who_picked_tomato', ''),
            'who_picked_plate': collaboration_data.get('who_picked_plate', ''),
            'who_cutted': collaboration_data.get('who_cutted', ''),
            'who_assembled': collaboration_data.get('who_assembled', ''),
            'who_delivered': collaboration_data.get('who_delivered', ''),
            'proportion_of_collaboration': collaboration_data.get('proportion_of_collaboration', [0.0, 0.0]),
            
            # Counter usage statistics
            'number_of_counters_used': total_counters_used,
            'number_of_counters_tomato': tomato_counter_count,
            'number_of_counters_plate': plate_counter_count,
            'number_of_counters_tomato_cut': tomato_cut_counter_count,
            'number_of_counters_tomato_salad': tomato_salad_counter_count,
            
            # Detailed counter usage history
            'history_of_counters_used': all_counter_history,
            'history_of_counters_tomato': tomato_counter_history,
            'history_of_counters_plate': plate_counter_history,
            'history_of_counters_tomato_cut': tomato_cut_counter_history,
            'history_of_counters_tomato_salad': tomato_salad_counter_history
        }
        
        # Add this action record to the agent's data
        agent_data[agent_id].append(action_data)
    
    # Now save each agent's data to CSV files
    for agent_id, action_data_list in agent_data.items():
        if action_data_list:
            # Create the action dataframe
            action_df = pd.DataFrame(action_data_list)
            
            # Save to CSV
            filename = f"{agent_id}_actions.csv"
            filepath = output_dir / filename
            action_df.to_csv(filepath, index=False)
            action_files[agent_id] = filepath
            
            print(f"Generated action file for {agent_id}: {filepath}")
            print(f"  Total actions: {len(action_df)}")
            print(f"  Total distance: {action_df['distance_walked'].iloc[-1] if not action_df.empty else 0:.2f} pixels")
            print(f"  Start position: {agent_tracking[agent_id]['start_pos']}")
            print(f"  Items touched: {len(set(action_df['touched_list'].iloc[-1].split(';')) if action_df['touched_list'].iloc[-1] else [])}")
            print(f"  Total counters used: {action_df['number_of_counters_used'].iloc[-1] if not action_df.empty else 0}")
            print(f"  Collaboration actions: {sum(action_df['is_item_collaboration'])}")
    
    # Print global collaboration summary
    total_salads = len([item_id for item_id in id_connection if item_id.startswith('tomato_salad_')])
    total_items = len(id_connection)
    collaborative_items = sum(1 for item_id in item_history if len(set(item_history[item_id].get('touched_list', []))) > 1)
    
    print(f"\nGlobal Analysis Summary:")
    print(f"  Total items tracked: {total_items}")
    print(f"  Total completed salads: {total_salads}")
    print(f"  Collaborative items: {collaborative_items}")
    print(f"  Items by type:")
    for item_type in ['tomato', 'plate', 'tomato_cut', 'tomato_salad']:
        # Count items if item_id == f'{item_type}_{Number}' with whatever number (e.g., tomato_1, tomato_2, ...)
        items_of_type = [
            item_id
            for item_id in id_connection
            if re.match(rf'^{re.escape(item_type)}_\d+$', item_id)
        ]        
        count = len(items_of_type)
        print(f"    {item_type}: {count} items: {items_of_type}")
    
    print(f"\nAll registered items (id_connection):")
    for item_id, connection_data in id_connection.items():
        history_data = item_history.get(item_id, {})
        action_data = item_actions.get(item_id, {})
        print(f"  {item_id}: connections={connection_data}, touched_list={history_data.get('touched_list', [])}, actions={action_data}")
    
    return action_files

def merge_actions_with_positions(meaningful_actions_df, positions_dir, output_dir=None, engine_tick_rate=24):
    """
    Merge meaningful action data with position data to create comprehensive action logs.
    
    Args:
        meaningful_actions_df: DataFrame with meaningful action data or path to meaningful_actions.csv file
        positions_dir: Directory containing position CSV files
        output_dir: Directory to save merged files (optional)
        engine_tick_rate: Engine tick rate from simulation config (default 24)
        
    Returns:
        Dictionary with merged action-position data per agent
    """
    
    if isinstance(meaningful_actions_df, (str, Path)):
        meaningful_actions_df = pd.read_csv(meaningful_actions_df)
    
    positions_dir = Path(positions_dir)
    merged_data = {}
    
    for agent_id in meaningful_actions_df['agent_id'].unique():
        if not agent_id.startswith('ai_rl_'):
            continue
            
        try:
            # Load position data for this agent
            position_file = positions_dir / f"{agent_id}_positions.csv"
            if not position_file.exists():
                print(f"Warning: Position file not found for {agent_id}: {position_file}")
                continue
                
            position_df = pd.read_csv(position_file)
            agent_actions = meaningful_actions_df[meaningful_actions_df['agent_id'] == agent_id].copy()
        
            # Merge on timing based on frame
            merged_actions = []
            for _, action in agent_actions.iterrows():
                frame = action.get('frame', 0)
                
                # Find corresponding position data by matching time
                # Position data should have a 'second' column, so convert frame to seconds
                action_second = frame / engine_tick_rate
                
                # Find closest position data by time
                if not position_df.empty and 'second' in position_df.columns:
                    # Find row with closest time
                    time_diffs = abs(position_df['second'] - action_second)
                    closest_idx = time_diffs.idxmin()
                    pos_row = position_df.loc[closest_idx]
                    
                    merged_action = {
                        **action.to_dict(),
                        'position_second': pos_row['second'],
                        'position_x': pos_row['x'],
                        'position_y': pos_row['y'],
                        'distance_walked': pos_row.get('distance_walked', 0),
                        'walking_speed': pos_row.get('walking_speed', 1.0),
                        'cutting_speed': pos_row.get('cutting_speed', 1.0),
                        'start_pos': pos_row.get('start_pos', '(0, 0)')
                    }
                    merged_actions.append(merged_action)
            
            merged_data[agent_id] = pd.DataFrame(merged_actions)
            
            # Save merged file if output directory is provided
            if output_dir:
                output_dir = Path(output_dir)
                filename = f"{agent_id}_actions_with_positions.csv"
                filepath = output_dir / filename
                merged_data[agent_id].to_csv(filepath, index=False)
                print(f"Saved merged action-position data for {agent_id}: {filepath}")
        
        except Exception as e:
            print(f"Error processing merged actions for agent {agent_id}: {e}")
            continue
    
    return merged_data