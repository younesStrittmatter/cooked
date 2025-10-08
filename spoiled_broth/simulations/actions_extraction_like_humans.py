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


def generate_agent_action_files(meaningful_actions_df, simulation_df, output_dir, map_name=None, simulation_id=None, engine_tick_rate=24, agent_initialization_period=15.0):
    """
    Generate action CSV files for each agent from meaningful_actions.csv and simulation data.
    
    Creates individual CSV files for each agent with comprehensive item tracking including:
    - Basic columns: second, item, item_id, action, target_type, target_position, action_long, 
      player_id, map_name, game_id, distance_walked, distance_walked_since_last_action, 
      overall_score, player_score_change, player_score, walking_speed, cutting_speed, start_pos
    - Item tracking: last_touched, touched_list, tomato_id, tomato_history, plate_id, plate_history, 
      tomato_cut_id, tomato_cut_history, tomato_salad_id, tomato_salad_history,
      is_exchange_collaboration,
    - Collaboration tracking: who_picked_tomato, who_picked_plate, who_cutted, who_assembled, 
      who_delivered, proportion_of_collaboration
    - Counter usage: number_of_counters_used, number_of_counters_tomato, number_of_counters_plate, 
      number_of_counters_tomato_cut, number_of_counters_tomato_salad, history_of_counters_used, 
      history_of_counters_tomato, history_of_counters_plate, history_of_counters_tomato_cut, 
      history_of_counters_tomato_salad
    
    The function filters out the initialization period and adjusts time so that
    the first second after initialization becomes second 0.
    
    Args:
        meaningful_actions_df: DataFrame with meaningful action data or path to meaningful_actions.csv file
        simulation_df: DataFrame with simulation data or path to simulation CSV file
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
    if isinstance(simulation_df, (str, Path)):
        simulation_df = pd.read_csv(simulation_df)
    
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
        
    # Sort dataframes
    meaningful_actions_df = meaningful_actions_df.sort_values(['agent_id', 'frame']).reset_index(drop=True)
    simulation_df = simulation_df.sort_values(['agent_id', 'frame']).reset_index(drop=True)
    
    # Filter out initialization period from meaningful actions
    # Convert frames to seconds to filter out initialization period
    meaningful_actions_df['temp_second'] = meaningful_actions_df['frame'] / engine_tick_rate
    meaningful_actions_df = meaningful_actions_df[meaningful_actions_df['temp_second'] >= agent_initialization_period].copy()
    meaningful_actions_df = meaningful_actions_df.drop(columns=['temp_second']).reset_index(drop=True)
    
    # Filter out initialization period from simulation data  
    simulation_df = simulation_df[simulation_df['second'] >= agent_initialization_period].copy()
    # Adjust time so that first second after initialization becomes second 0
    simulation_df['second'] = simulation_df['second'] - agent_initialization_period
    simulation_df = simulation_df.reset_index(drop=True)
    
    # Also need to adjust frame numbers in meaningful_actions_df to match the time adjustment
    meaningful_actions_df['adjusted_frame'] = (meaningful_actions_df['frame'] / engine_tick_rate - agent_initialization_period) * engine_tick_rate
    meaningful_actions_df['frame'] = meaningful_actions_df['adjusted_frame'].astype(int)
    meaningful_actions_df = meaningful_actions_df.drop(columns=['adjusted_frame'])
    
    if meaningful_actions_df.empty:
        print(f"Warning: No meaningful actions found after filtering initialization period of {agent_initialization_period} seconds")
        return {}
    
    print(f"Filtered out initialization period of {agent_initialization_period} seconds")
    print(f"Remaining meaningful actions: {len(meaningful_actions_df)}")
    print(f"Remaining simulation data points: {len(simulation_df)}")
    
    # Use engine tick rate from simulation config
    ENGINE_TICK_RATE = engine_tick_rate
    
    # Item tracking data structures - track individual items by ID
    item_registry = {}  # item_id -> {type, origin_ids, touched_list, counter_history, etc.}
    item_index = {'tomato': 1, 'plate': 1, 'tomato_cut': 1, 'tomato_salad': 1}  # Item ID counters
    
    # Action role tracking
    role_actions = {
        'who_picked_tomato': {},    # tomato_id -> agent_id
        'who_picked_plate': {},     # plate_id -> agent_id
        'who_cutted': {},           # tomato_cut_id -> agent_id
        'who_assembled': {},        # tomato_salad_id -> agent_id
        'who_delivered': {}         # tomato_salad_id -> agent_id
    }
    
    # Track cutting actions - map agent_id to tomato_id being cut
    cutting_registry = {}  # agent_id -> tomato_id (currently being cut)
    
    def create_item_id(item_type):
        """Create unique item identifier and increment counter."""
        item_id = f"{item_type}_{item_index[item_type]}"
        print(f"DEBUG: Creating item ID: {item_id} (counter before increment: {item_index[item_type]})")
        item_index[item_type] += 1
        return item_id
    
    def register_new_item(item_id, item_type, agent_id, frame, origin_ids=None, engine_tick_rate=24):
        """Register a new item in the tracking system."""
        item_registry[item_id] = {
            'type': item_type,
            'created_by': agent_id,
            'created_frame': frame,
            'touched_list': [agent_id],  # Most recent first
            'last_touched': agent_id,
            'origin_ids': origin_ids or {},  # {'tomato_id': 'tomato_1', 'plate_id': 'plate_2'}
            'counter_history': [],  # List of (agent_id, action_type, frame) for counter interactions
            'counter_count': 0
        }
        second = frame / engine_tick_rate
        print(f"Created new item: {item_id} (type: {item_type}) by {agent_id} at {second:.2f}s (frame {frame})")
        if origin_ids:
            print(f"  Origin items: {origin_ids}")
        return item_registry[item_id]
    
    def update_item_interaction(item_id, agent_id, frame, action_type, target_type=None, target_x=None, target_y=None):
        """Update item interaction tracking."""
        if item_id not in item_registry:
            return None
            
        item_data = item_registry[item_id]
        
        # Update touched list - ensure agent is recorded when they interact with item
        if agent_id not in item_data['touched_list'] or agent_id != item_data['last_touched']:
            if agent_id not in item_data['touched_list']:
                item_data['touched_list'].append(agent_id)
                print(f"  Added {agent_id} to touched_list for {item_id}. New touched_list: {item_data['touched_list']}")
            item_data['last_touched'] = agent_id
        
        # Track counter interactions (place/pickup)
        if action_type in ['drop', 'pickup'] and target_type == 'counter':
            item_data['counter_history'].append((agent_id, action_type, frame, target_x, target_y))
            if action_type == 'drop':
                item_data['counter_count'] += 1
        
        return item_data
    
    # Global tracking of items by location
    counter_items = {}  # (x, y) -> {item_id, type, second_dropped}
    
    def get_current_item_id(current_item_name, agent_id, frame, action_type, target_type, target_x=None, target_y=None, engine_tick_rate=24):
        """Determine the current item ID based on action and item names. Track items by location."""
        if not current_item_name or current_item_name == 'None':
            return None
            
        # Map item names to types
        item_name_lower = current_item_name
        action_lower = action_type
        target_type = target_type

        if action_category_name == 'assemble salad':
            # When assembling salad, agent drops tomato_cut or plate on counter
            # The salad is created on the counter, not in agent's hand
            # So we return the ID of the item being dropped
            if item_name_lower in ['tomato_cut', 'plate']:
                # Create the salad on the counter but don't return it as current item
                salad_id = create_item_id('tomato_salad')
                origin_ids = find_origin_items_for_salad(current_item_name, salad_id, target_x, target_y)
                register_new_item(salad_id, 'tomato_salad', agent_id, frame, origin_ids, engine_tick_rate)
                role_actions['who_assembled'][salad_id] = agent_id
                
                # Store the salad on the counter
                if target_x is not None and target_y is not None:
                    location = (target_x, target_y)
                    counter_items[location] = {
                        'item_id': salad_id,
                        'type': 'tomato_salad',
                        'second_dropped': frame / engine_tick_rate
                    }
                
                # Return None because agent's hand becomes empty after assembly
                return None
                        
        elif item_name_lower == 'tomato_cut':
            # Create new tomato_cut when picking up from cutting board after cutting
            if (action_lower == 'pickup' and target_type == 'cuttingboard'):
                item_id = create_item_id('tomato_cut')
                origin_tomato_id = cutting_registry.get(agent_id)
                origin_ids = {'tomato_id': origin_tomato_id} if origin_tomato_id else {}
                register_new_item(item_id, 'tomato_cut', agent_id, frame, origin_ids, engine_tick_rate)
                role_actions['who_cutted'][item_id] = agent_id
                if agent_id in cutting_registry:
                    del cutting_registry[agent_id]
                return item_id
            # Handle tomato_cut pickup from counter (by different agent)
            elif (action_lower == 'pickup' and target_type == 'counter' and target_x is not None and target_y is not None):
                location = (target_x, target_y)
                if location in counter_items and counter_items[location]['type'] == 'tomato_cut':
                    item_id = counter_items[location]['item_id']
                    # Update the item interaction to record that this agent is now handling it
                    update_item_interaction(item_id, agent_id, frame, action_type, target_type, target_x, target_y)
                    return item_id
                        
        elif item_name_lower == 'tomato_salad':
            # Handle tomato_salad pickup from counter
            if (action_lower == 'pickup' and target_type == 'counter' and target_x is not None and target_y is not None):
                location = (target_x, target_y)
                if location in counter_items and counter_items[location]['type'] == 'tomato_salad':
                    item_id = counter_items[location]['item_id']
                    # Update the item interaction to record that this agent is now handling it
                    update_item_interaction(item_id, agent_id, frame, action_type, target_type, target_x, target_y)
                    return item_id
                        
        else:
            # Create new tomato/plate when picked from dispenser
            if (action_lower == 'pickup' and target_type == 'dispenser'):
                item_id = create_item_id(item_name_lower)
                register_new_item(item_id, item_name_lower, agent_id, frame, None, engine_tick_rate)
                role_actions[f'who_picked_{item_name_lower}'][item_id] = agent_id
                return item_id
            elif (action_lower == 'pickup' and target_type == 'counter' and target_x is not None and target_y is not None):
                location = (target_x, target_y)
                if location in counter_items and counter_items[location]['type'] == item_name_lower:
                    item_id = counter_items[location]['item_id']
                    # Update the item interaction to record that this agent is now handling it
                    update_item_interaction(item_id, agent_id, frame, action_type, target_type, target_x, target_y)
                    return item_id
            else:
                print(f"Warning: Unable to determine {item_name_lower} item_id for action '{action_type}' with target_type '{target_type}'")
                        
        return None
    
    def track_item_placement(item_id, target_x, target_y, frame, action_type, target_type, engine_tick_rate):
        """Track when items are dropped on counters."""
        if item_id and target_x is not None and target_y is not None:
            location = (target_x, target_y)
            if action_type == 'drop' and target_type == 'counter':
                if item_id in item_registry:
                    item_type = item_registry[item_id]['type']
                    counter_items[location] = {
                        'item_id': item_id,
                        'type': item_type,
                        'second_dropped': frame / engine_tick_rate
                    }

            elif action_type == 'pickup' and target_type == 'counter':
                # Remove item from counter tracking when picked up
                if location in counter_items and counter_items[location]['item_id'] == item_id:
                    del counter_items[location]
    
    def find_origin_items_for_salad(current_item_name=None, salad_id=None, target_x=None, target_y=None):
        """Find the tomato_cut and plate that were combined to make a salad."""
        origin_ids = {}
        
        # New logic: determine what's being dropped and what's already on the counter
        if current_item_name and target_x is not None and target_y is not None:
            location = (target_x, target_y)
            item_name_lower = current_item_name
            
            if item_name_lower == 'plate':
                # If dropping a plate, we need to find the plate ID from the agent's tracking
                # and look for tomato_cut on counter
                for agent_id, agent_items in last_items.items():
                    if agent_items.get('plate'):
                        origin_ids['plate_id'] = agent_items['plate']
                        break
                                
                # Look for tomato_cut on the counter at this location
                if location in counter_items and counter_items[location]['type'] == 'tomato_cut':
                    tomato_cut_id = counter_items[location]['item_id']
                    origin_ids['tomato_cut_id'] = tomato_cut_id
                    # Get the tomato origin from the tomato_cut
                    if tomato_cut_id in item_registry:
                        tomato_cut_data = item_registry[tomato_cut_id]
                        if 'tomato_id' in tomato_cut_data.get('origin_ids', {}):
                            origin_ids['tomato_id'] = tomato_cut_data['origin_ids']['tomato_id']
                            
            elif item_name_lower == 'tomato_cut':
                # If dropping a tomato_cut, find its ID from agent's tracking
                # and look for plate on counter
                for agent_id, agent_items in last_items.items():
                    if agent_items.get('tomato_cut'):
                        tomato_cut_id = agent_items['tomato_cut']
                        origin_ids['tomato_cut_id'] = tomato_cut_id
                        if tomato_cut_id in item_registry and 'tomato_id' in item_registry[tomato_cut_id].get('origin_ids', {}):
                            origin_ids['tomato_id'] = item_registry[tomato_cut_id]['origin_ids']['tomato_id']
                        break

                # Look for plate on the counter at this location
                if location in counter_items and counter_items[location]['type'] == 'plate':
                    origin_ids['plate_id'] = counter_items[location]['item_id']
        
        # Ensure we have complete origin information by searching all available items if needed
        if not origin_ids.get('tomato_id') and origin_ids.get('tomato_cut_id'):
            tomato_cut_id = origin_ids['tomato_cut_id']
            if tomato_cut_id in item_registry:
                tomato_cut_data = item_registry[tomato_cut_id]
                if 'tomato_id' in tomato_cut_data.get('origin_ids', {}):
                    origin_ids['tomato_id'] = tomato_cut_data['origin_ids']['tomato_id']
        
        return origin_ids
    
    def get_item_collaboration_data(item_id):
        """Get collaboration data for a specific item."""
        if item_id not in item_registry:
            return {}, False, False
            
        item_data = item_registry[item_id]
        
        # Check if multiple agents touched this specific item
        is_item_collaboration = len(set(item_data['touched_list'])) > 1
        
        # Check if multiple agents were involved in the full history
        all_involved_agents = set(item_data['touched_list'])
        
        # Add agents from origin items
        for origin_id in item_data.get('origin_ids', {}).values():
            if origin_id in item_registry:
                all_involved_agents.update(item_registry[origin_id]['touched_list'])
        
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
        
        # Get role assignments based on item history
        if item_data['type'] == 'tomato_salad':
            salad_id = item_id
            tomato_cut_id = item_data.get('origin_ids', {}).get('tomato_cut_id')
            tomato_id = item_data.get('origin_ids', {}).get('tomato_id')
            plate_id = item_data.get('origin_ids', {}).get('plate_id')
            
            collaboration_data['who_assembled'] = role_actions['who_assembled'].get(salad_id, '')
            collaboration_data['who_cutted'] = role_actions['who_cutted'].get(tomato_cut_id, '')
            collaboration_data['who_picked_tomato'] = role_actions['who_picked_tomato'].get(tomato_id, '')
            collaboration_data['who_picked_plate'] = role_actions['who_picked_plate'].get(plate_id, '')
            collaboration_data['who_delivered'] = role_actions['who_delivered'].get(salad_id, '')
            
            # Calculate percentages (each role = 0.2, total = 1.0)
            agent_contributions = {}
            role_weight = 0.2
            for role, agent in collaboration_data.items():
                if agent and role != 'proportion_of_collaboration':
                    agent_contributions[agent] = agent_contributions.get(agent, 0) + role_weight
            
            # Convert to list format [agent1_contribution, agent2_contribution]
            agents = sorted(list(all_involved_agents))  # Sort for consistency
            if len(agents) >= 2:
                collaboration_data['proportion_of_collaboration'] = [
                    agent_contributions.get(agents[0], 0.0),
                    agent_contributions.get(agents[1], 0.0)
                ]
            elif len(agents) == 1:
                collaboration_data['proportion_of_collaboration'] = [
                    agent_contributions.get(agents[0], 0.0),
                    0.0
                ]
        
        elif item_data['type'] == 'tomato_cut':
            tomato_cut_id = item_id
            tomato_id = item_data.get('origin_ids', {}).get('tomato_id')
            collaboration_data['who_cutted'] = role_actions['who_cutted'].get(tomato_cut_id, '')
            collaboration_data['who_picked_tomato'] = role_actions['who_picked_tomato'].get(tomato_id, '')
            
            # Calculate percentages for tomato_cut (2 roles = 0.5 each)
            agent_contributions = {}
            role_weight = 0.5
            for role in ['who_picked_tomato', 'who_cutted']:
                agent = collaboration_data[role]
                if agent:
                    agent_contributions[agent] = agent_contributions.get(agent, 0) + role_weight
            
            agents = sorted(list(all_involved_agents))
            if len(agents) >= 2:
                collaboration_data['proportion_of_collaboration'] = [
                    agent_contributions.get(agents[0], 0.0),
                    agent_contributions.get(agents[1], 0.0)
                ]
            elif len(agents) == 1:
                collaboration_data['proportion_of_collaboration'] = [
                    agent_contributions.get(agents[0], 0.0),
                    0.0
                ]

        else:         
            collaboration_data[f'who_picked_{item_data["type"]}'] = role_actions[f'who_picked_{item_data["type"]}'].get(item_id, '')
            # For basic items (tomato, plate), no collaboration calculation needed
            agents = sorted(list(all_involved_agents))
            if len(agents) >= 1:
                collaboration_data['proportion_of_collaboration'] = [1.0, 0.0] if len(agents) == 1 else [0.5, 0.5]

        return collaboration_data, is_item_collaboration, is_exchange_collaboration

    # Get histories for each item type
    def get_item_history(item_id):
        if item_id and item_id in item_registry:
            return ';'.join(item_registry[item_id].get('touched_list', []))
        return ''
        
    # Get counter usage data for the current item
    def get_item_counter_data(item_id, engine_tick_rate):
        if item_id and item_id in item_registry:
            counter_history = item_registry[item_id].get('counter_history', [])
            counter_count = item_registry[item_id].get('counter_count', 0)
            # Convert frames to seconds and create history string
            history_str = ';'.join([f"{agent}:{action}@{frame/engine_tick_rate:.2f}s" for agent, action, frame, target_x, target_y in counter_history])
            return counter_count, history_str
        return 0, ''
    
    # Get all counter history entries sorted by time
    def get_all_counter_history_sorted(item_ids, engine_tick_rate):
        all_entries = []
        for item_id in item_ids:
            if item_id and item_id in item_registry:
                counter_history = item_registry[item_id].get('counter_history', [])
                for agent, action, frame, target_x, target_y in counter_history:
                    all_entries.append({
                        'frame': frame,
                        'second': frame / engine_tick_rate,
                        'agent': agent,
                        'action': action,
                        'item_id': item_id
                    })
        
        # Sort by frame number
        all_entries.sort(key=lambda x: x['frame'])
        
        # Create formatted history string
        history_str = ';'.join([f"{entry['agent']}:{entry['action']}@{entry['second']:.2f}s" for entry in all_entries])
        return history_str
    
    # Load position data for distance calculations
    position_files = {}
    for agent_id in simulation_df['agent_id'].unique():
        if agent_id.startswith('ai_rl_'):
            position_file = output_dir / f"{agent_id}_positions.csv"
            if position_file.exists():
                position_files[agent_id] = pd.read_csv(position_file)
            else:
                # Generate position data on the fly if file doesn't exist
                agent_sim_data = simulation_df[simulation_df['agent_id'] == agent_id].copy()
                agent_sim_data = agent_sim_data.sort_values('frame').reset_index(drop=True)
                
                # Calculate distance walked
                agent_sim_data['distance_walked'] = 0.0
                if len(agent_sim_data) > 1:
                    for i in range(1, len(agent_sim_data)):
                        prev_x = agent_sim_data.loc[i-1, 'x']
                        prev_y = agent_sim_data.loc[i-1, 'y']
                        curr_x = agent_sim_data.loc[i, 'x']
                        curr_y = agent_sim_data.loc[i, 'y']
                        distance_step = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                        agent_sim_data.loc[i, 'distance_walked'] = agent_sim_data.loc[i-1, 'distance_walked'] + distance_step
                
                position_files[agent_id] = agent_sim_data[['second', 'x', 'y', 'distance_walked']].copy()
    
    # Initialize agent-specific data structures for CSV writing
    agent_data = {}  # agent_id -> list of action records
    agent_tracking = {}  # agent_id -> tracking variables (last distance, score, etc.)
    
    # Initialize tracking for each agent
    for agent_id in meaningful_actions_df['agent_id'].unique():
        if not agent_id.startswith('ai_rl_'):
            continue
            
        agent_sim_data = simulation_df[simulation_df['agent_id'] == agent_id].copy()
        if agent_sim_data.empty:
            continue
            
        # Get start position from simulation data at frame 0
        start_frame_data = agent_sim_data[agent_sim_data['frame'] == 0]
        if not start_frame_data.empty:
            start_tile_x = start_frame_data.iloc[0]['tile_x']
            start_tile_y = start_frame_data.iloc[0]['tile_y']
            start_pos = f"({start_tile_x}, {start_tile_y})"
        else:
            start_tile_x = agent_sim_data.iloc[0]['tile_x']
            start_tile_y = agent_sim_data.iloc[0]['tile_y']
            start_pos = f"({start_tile_x}, {start_tile_y})"
        
        # Calculate max distance for this agent
        max_distance = 0.0
        position_data = position_files.get(agent_id, pd.DataFrame())
        if not position_data.empty:
            max_distance = position_data['distance_walked'].max()
        elif not agent_sim_data.empty:
            # Calculate from simulation data
            for i in range(1, len(agent_sim_data)):
                prev_x = agent_sim_data.iloc[i-1]['x']
                prev_y = agent_sim_data.iloc[i-1]['y']
                curr_x = agent_sim_data.iloc[i]['x']
                curr_y = agent_sim_data.iloc[i]['y']
                max_distance += math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        
        agent_data[agent_id] = []
        agent_tracking[agent_id] = {
            'previous_distance': 0.0,
            'previous_player_score': 0,
            'last_touched_item': 'None',
            'touched_items': [],
            'start_pos': start_pos,
            'max_distance': max_distance,
            'position_data': position_data,
            'sim_data': agent_sim_data
        }
    
    # Sort all meaningful actions across all agents by frame to process chronologically
    all_actions = meaningful_actions_df.sort_values(['frame', 'agent_id']).reset_index(drop=True)
    last_items = {}  # agent_id -> {item_type: item_id} to track different item types separately

    # Process all actions chronologically
    for _, action in all_actions.iterrows():
        agent_id = action.get('agent_id')
        frame = action.get('frame', 0)
        action_category_name = action.get('action_category_name', '')
        current_item = action.get('current_item', None)
        previous_item = action.get('previous_item', None)
        action_type = action.get('item_change_type', '')
        target_type = action.get('target_tile_type', '')
        target_x = action.get('target_tile_x', None)
        target_y = action.get('target_tile_y', None)
        
        # Skip non-RL agents
        if not agent_id.startswith('ai_rl_') or agent_id not in agent_tracking:
            print(f"DEBUG: Skipping action from agent {agent_id} (frame {frame}): {action_category_name}")
            continue
        
        # Calculate timing from frame and ENGINE_TICK_RATE
        second = frame / ENGINE_TICK_RATE
        
        # Get or create item ID based on action type
        item_involved = current_item if current_item and current_item != 'None' else previous_item

        # Initialize agent's item tracking if not exists
        if agent_id not in last_items:
            last_items[agent_id] = {}

        # Special handling for cutting actions - when we start cutting, we're working with a tomato
        if action_type == 'cutting':
            # Find the tomato that was just placed on the cutting board and register it
            item_involved = 'tomato'
            tomato_id = last_items[agent_id].get('tomato')
            cutting_registry[agent_id] = tomato_id
            if tomato_id:
                update_item_interaction(tomato_id, agent_id, frame, action_type)

        # Determine current item ID based on item type and existing tracking
        current_item_id = None
        if item_involved and item_involved != 'None':
            # Check if we need to create a new item or use existing one
            existing_item_id = last_items[agent_id].get(item_involved)
            
            # Create new item if:
            # 1. No existing item of this type for this agent
            # 2. Action involves picking up from dispenser (new item)
            # 3. Action is assembling salad (creates new tomato_salad but agent hand becomes empty)
            # 4. Action is picking up tomato_cut from cutting board (new tomato_cut)
            # 5. Action is picking up from counter (existing item)
            if (existing_item_id is None or 
                (action_type == 'pickup' and target_type == 'dispenser') or
                action_category_name == 'assemble salad' or
                (action_type == 'pickup' and target_type == 'cuttingboard' and item_involved == 'tomato_cut') or
                (action_type == 'pickup' and target_type == 'counter')):
                
                current_item_id = get_current_item_id(item_involved, agent_id, frame, action_type, target_type, target_x, target_y, ENGINE_TICK_RATE)
                if current_item_id:
                    last_items[agent_id][item_involved] = current_item_id
                elif action_category_name == 'assemble salad':
                    # For salad assembly, clear the item being dropped since hand becomes empty
                    if item_involved in last_items[agent_id]:
                        last_items[agent_id][item_involved] = None
            else:
                current_item_id = existing_item_id

        # For salad assembly actions, we need to track the item being dropped, not the salad
        if action_category_name == 'assemble salad':
            # The item_involved is what the agent was holding (tomato_cut or plate)
            # Get the current item ID before it gets cleared
            if agent_id in last_items and item_involved in last_items[agent_id]:
                current_item_id = last_items[agent_id][item_involved]

        # Update item tracking if we have a valid item ID
        if current_item_id:
            print(f"Frame {frame}: Agent {agent_id} performing {action_type} on {current_item_id} (target: {target_type})")
            update_item_interaction(current_item_id, agent_id, frame, action_type, target_type, target_x, target_y)
            track_item_placement(current_item_id, target_x, target_y, frame, action_type, target_type, ENGINE_TICK_RATE)

            # Update agent's item tracking based on action type
            item_type = current_item_id.split('_')[0]
            if 'cut' in current_item_id:
                item_type = 'tomato_cut'
            elif 'salad' in current_item_id:
                item_type = 'tomato_salad'
            
            if action_type == 'pickup':
                # Agent now has this item
                last_items[agent_id][item_type] = current_item_id
            elif action_type == 'drop':
                # Check where the item is being dropped
                if target_type in ['counter', 'cuttingboard']:
                    # Item is placed, agent no longer holds it
                    last_items[agent_id][item_type] = None
                elif target_type == 'delivery':
                    # Track delivery actions for salads
                    if current_item_id.startswith('tomato_salad_'):
                        role_actions['who_delivered'][current_item_id] = agent_id
                    # Delivered items are no longer held
                    last_items[agent_id][item_type] = None
        
        # Special handling for salad assembly - clear the item being assembled from agent's tracking
        if action_category_name == 'assemble salad' and item_involved in ['tomato_cut', 'plate']:
            if agent_id in last_items and item_involved in last_items[agent_id]:
                last_items[agent_id][item_involved] = None
        
        # Get agent tracking data
        tracking = agent_tracking[agent_id]
        if current_item_id != tracking['last_touched_item']:
            tracking['last_touched_item'] = current_item_id
            if current_item_id not in tracking['touched_items']:
                tracking['touched_items'].append(current_item_id)

        # Find corresponding simulation data to get score and distance
        sim_frame_data = tracking['sim_data'][tracking['sim_data']['frame'] == frame]
        if not sim_frame_data.empty:
            sim_row = sim_frame_data.iloc[0]
            current_player_score = sim_row.get('score', 0)
            
            # Calculate overall score as sum of all players' scores at this frame
            frame_data_all_agents = simulation_df[simulation_df['frame'] == frame]
            overall_score = frame_data_all_agents['score'].sum() if not frame_data_all_agents.empty else current_player_score
            
            # Calculate distance walked at this frame
            if not tracking['position_data'].empty:
                # Find position data for this frame
                pos_data_for_frame = tracking['position_data'][tracking['position_data']['second'] <= second]
                if not pos_data_for_frame.empty:
                    distance_walked = pos_data_for_frame.iloc[-1]['distance_walked']
                else:
                    distance_walked = 0.0
            else:
                # Approximate based on frame progression
                total_frames = tracking['sim_data']['frame'].max() if not tracking['sim_data'].empty else 1
                distance_walked = (frame / total_frames) * tracking['max_distance']
        else:
            print(f"Warning: No simulation data found for agent {agent_id} at frame {frame}")
            # Use nearest simulation data
            if not tracking['sim_data'].empty:
                closest_sim = tracking['sim_data'].iloc[(tracking['sim_data']['frame'] - frame).abs().argsort()[:1]]
                if not closest_sim.empty:
                    current_player_score = closest_sim.iloc[0].get('score', 0)
                    # Try to calculate overall score for the closest frame
                    closest_frame = closest_sim.iloc[0]['frame']
                    frame_data_all_agents = simulation_df[simulation_df['frame'] == closest_frame]
                    overall_score = frame_data_all_agents['score'].sum() if not frame_data_all_agents.empty else current_player_score
                else:
                    current_player_score = 0
                    overall_score = 0
            else:
                current_player_score = 0
                overall_score = 0
            distance_walked = 0.0
        
        # Calculate distance walked since last action
        distance_walked_since_last_action = distance_walked - tracking['previous_distance']
        tracking['previous_distance'] = distance_walked
        
        # Calculate player score change (difference from previous frame)
        player_score_change = current_player_score - tracking['previous_player_score']
        tracking['previous_player_score'] = current_player_score
        
        # Create target_position in the format (x_tile, y_tile)
        target_position = f"({target_x}, {target_y})"
        
        # Get item-specific data if we have a current item ID
        item_data = {}
        collaboration_data = {}
        is_item_collaboration = False
        is_exchange_collaboration = False
        
        if current_item_id and current_item_id in item_registry:
            item_data = item_registry[current_item_id]
            collaboration_data, is_item_collaboration, is_exchange_collaboration = get_item_collaboration_data(current_item_id)
        
        # Extract item IDs and histories based on current item
        last_touched = item_data.get('last_touched', '')
        touched_list_str = ';'.join(item_data.get('touched_list', [])) if item_data.get('touched_list') else ''
        
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
        if current_item_id and current_item_id in item_registry:
            item_data = item_registry[current_item_id]
            
            if current_item_id.startswith('tomato_salad_'):
                # For tomato_salad: populate all related items
                tomato_salad_id = current_item_id
                tomato_salad_history = get_item_history(current_item_id)
                
                origin_ids = item_data.get('origin_ids', {})
                # Get tomato info from origin
                if origin_ids.get('tomato_id'):
                    tomato_id = origin_ids.get('tomato_id', '')
                    tomato_history = get_item_history(tomato_id)
                # Get plate info from origin
                if origin_ids.get('plate_id'):
                    plate_id = origin_ids.get('plate_id', '')
                    plate_history = get_item_history(plate_id)
                # Get tomato_cut info from origin
                if origin_ids.get('tomato_cut_id'):
                    tomato_cut_id = origin_ids.get('tomato_cut_id', '')
                    tomato_cut_history = get_item_history(tomato_cut_id)
                    
            elif current_item_id.startswith('tomato_cut_'):
                # For tomato_cut: populate tomato_cut and its origin tomato
                tomato_cut_id = current_item_id
                tomato_cut_history = get_item_history(current_item_id)
                
                origin_ids = item_data.get('origin_ids', {})
                if origin_ids.get('tomato_id'):
                    tomato_id = origin_ids.get('tomato_id', '')
                    tomato_history = get_item_history(tomato_id)
                    
            elif current_item_id.startswith('tomato_'):
                # For tomato: only populate tomato info
                tomato_id = current_item_id
                tomato_history = get_item_history(current_item_id)
                
            elif current_item_id.startswith('plate_'):
                # For plate: only populate plate info
                plate_id = current_item_id
                plate_history = get_item_history(current_item_id)
        
        # Additional check: if we're missing information and this is a pickup action from counter,
        # we might be picking up an item with full history that we should preserve
        elif not current_item_id and action_type == 'pickup' and target_type == 'counter' and target_x is not None and target_y is not None:
            # Check if there's an item on the counter at this location
            location = (target_x, target_y)
            if location in counter_items:
                counter_item = counter_items[location]
                potential_item_id = counter_item['item_id']
                
                if potential_item_id in item_registry:
                    current_item_id = potential_item_id
                    item_data = item_registry[current_item_id]
                    
                    # Update the agent's item tracking
                    item_type = item_data['type']
                    if agent_id not in last_items:
                        last_items[agent_id] = {}
                    last_items[agent_id][item_type] = current_item_id
                    
                    # Update interaction tracking
                    update_item_interaction(current_item_id, agent_id, frame, action_type, target_type, target_x, target_y)
                    
                    # Populate all related information based on item type
                    if current_item_id.startswith('tomato_salad_'):
                        tomato_salad_id = current_item_id
                        tomato_salad_history = get_item_history(current_item_id)
                        
                        origin_ids = item_data.get('origin_ids', {})
                        if origin_ids.get('tomato_id'):
                            tomato_id = origin_ids.get('tomato_id', '')
                            tomato_history = get_item_history(tomato_id)
                        if origin_ids.get('plate_id'):
                            plate_id = origin_ids.get('plate_id', '')
                            plate_history = get_item_history(plate_id)
                        if origin_ids.get('tomato_cut_id'):
                            tomato_cut_id = origin_ids.get('tomato_cut_id', '')
                            tomato_cut_history = get_item_history(tomato_cut_id)
                            
                    elif current_item_id.startswith('tomato_cut_'):
                        tomato_cut_id = current_item_id
                        tomato_cut_history = get_item_history(current_item_id)
                        
                        origin_ids = item_data.get('origin_ids', {})
                        if origin_ids.get('tomato_id'):
                            tomato_id = origin_ids.get('tomato_id', '')
                            tomato_history = get_item_history(tomato_id)
                    
                    elif current_item_id.startswith('tomato_'):
                        tomato_id = current_item_id
                        tomato_history = get_item_history(current_item_id)
                        
                    elif current_item_id.startswith('plate_'):
                        plate_id = current_item_id
                        plate_history = get_item_history(current_item_id)
        
        # Get counter data for all relevant items
        tomato_counter_count, tomato_counter_history = get_item_counter_data(tomato_id, ENGINE_TICK_RATE)
        plate_counter_count, plate_counter_history = get_item_counter_data(plate_id, ENGINE_TICK_RATE)
        tomato_cut_counter_count, tomato_cut_counter_history = get_item_counter_data(tomato_cut_id, ENGINE_TICK_RATE)
        tomato_salad_counter_count, tomato_salad_counter_history = get_item_counter_data(tomato_salad_id, ENGINE_TICK_RATE)
        
        total_counters_used = tomato_counter_count + plate_counter_count + tomato_cut_counter_count + tomato_salad_counter_count
        
        # Get all counter history sorted by frame/time for all relevant items
        relevant_item_ids = [tomato_id, plate_id, tomato_cut_id, tomato_salad_id]
        all_counter_history = get_all_counter_history_sorted(relevant_item_ids, ENGINE_TICK_RATE)
        
        action_data = {
            # Basic action data
            'second': second,
            'item': item_involved,
            'item_id': current_item_id,
            'action': action_type,
            'target_type': target_type,
            'target_position': target_position,
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
        print(f"DEBUG: Added action record for {agent_id} at frame {frame}: item_id={current_item_id}, action={action_type}")
    
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
            print(f"  Collaboration actions: {sum(action_df['is_exchange_collaboration'])}")
    
    # Print global collaboration summary
    total_salads = len([item_id for item_id in item_registry if item_id.startswith('tomato_salad_')])
    total_items = len(item_registry)
    collaborative_items = sum(1 for item_data in item_registry.values() if len(set(item_data.get('touched_list', []))) > 1)
    
    print(f"\nGlobal Analysis Summary:")
    print(f"  Total items tracked: {total_items}")
    print(f"  Total completed salads: {total_salads}")
    print(f"  Collaborative items: {collaborative_items}")
    print(f"  Items by type:")
    for item_type in ['tomato', 'plate', 'tomato_cut', 'tomato_salad']:
        # Count items if item_id == f'{item_type}_{Number}' with whatever number (e.g., tomato_1, tomato_2, ...)
        items_of_type = [
            item_id
            for item_id in item_registry
            if re.match(rf'^{re.escape(item_type)}_\d+$', item_id)
        ]        
        count = len(items_of_type)
        print(f"    {item_type}: {count} items: {items_of_type}")
    
    print(f"\nAll registered items:")
    for item_id, item_data in item_registry.items():
        print(f"  {item_id}: type={item_data['type']}, created_by={item_data['created_by']}, touched_list={item_data['touched_list']}")
    
    return action_files


def extract_detailed_action_sequences(meaningful_actions_df, simulation_df, output_dir=None, save_individual=True):
    """
    Extract detailed action sequences with timing, positioning, and state information.
    
    Args:
        meaningful_actions_df: DataFrame with meaningful action data or path to meaningful_actions.csv file
        simulation_df: DataFrame with simulation data or path to simulation CSV file
        output_dir: Directory to save files (optional)
        save_individual: Whether to save individual agent files
        
    Returns:
        Dictionary containing action sequences per agent
    """
    
    # Read the CSV files if they are paths
    if isinstance(meaningful_actions_df, (str, Path)):
        meaningful_actions_df = pd.read_csv(meaningful_actions_df)
    if isinstance(simulation_df, (str, Path)):
        simulation_df = pd.read_csv(simulation_df)
    
    action_sequences = {}
    
    # Process each agent
    for agent_id in meaningful_actions_df['agent_id'].unique():
        if not agent_id.startswith('ai_rl_'):
            continue
            
        try:
            agent_actions = meaningful_actions_df[meaningful_actions_df['agent_id'] == agent_id].copy()
            agent_sim_data = simulation_df[simulation_df['agent_id'] == agent_id].copy()
            
            # Sort by frame to get chronological order
            agent_actions = agent_actions.sort_values('frame').reset_index(drop=True)
            agent_sim_data = agent_sim_data.sort_values('frame').reset_index(drop=True)
            
            # Merge action data with simulation states
            detailed_actions = []
        
            for _, action in agent_actions.iterrows():
                # Use the frame from meaningful_actions directly
                frame = action.get('frame', 0)
                
                # Find simulation row for this frame
                sim_frame_data = agent_sim_data[agent_sim_data['frame'] == frame]
                if not sim_frame_data.empty:
                    sim_row = sim_frame_data.iloc[0]
                else:
                    sim_row = agent_sim_data.iloc[-1] if not agent_sim_data.empty else None
                
                if sim_row is not None:
                    detailed_action = {
                        'agent_id': agent_id,
                        'frame': action.get('frame'),
                        'action_id': action.get('action_id'),
                        'action_number': action.get('action_number'),
                        'action_type': action.get('action_type'),
                        'action_category_name': action.get('action_category_name'),
                        'target_tile_type': action.get('target_tile_type'),
                        'target_tile_x': action.get('target_tile_x'),
                        'target_tile_y': action.get('target_tile_y'),
                        'agent_tile_x': action.get('agent_tile_x'),
                        'agent_tile_y': action.get('agent_tile_y'),
                        'simulation_frame': sim_row['frame'],
                        'simulation_second': sim_row['second'],
                        'agent_x': sim_row['x'],
                        'agent_y': sim_row['y'],
                        'agent_tile_x_sim': sim_row['tile_x'],
                        'agent_tile_y_sim': sim_row['tile_y'],
                        'agent_item': sim_row.get('item', 'None'),
                        'agent_score': sim_row.get('score', 0),
                        'previous_item': action.get('previous_item'),
                        'current_item': action.get('current_item'),
                        'item_change_type': action.get('item_change_type'),
                        'compound_action_part': action.get('compound_action_part', 0)
                    }
                    detailed_actions.append(detailed_action)
        
            action_sequences[agent_id] = pd.DataFrame(detailed_actions)
        
            # Save individual file if requested
            if save_individual and output_dir:
                output_dir = Path(output_dir)
                filename = f"{agent_id}_detailed_actions.csv"
                filepath = output_dir / filename
                action_sequences[agent_id].to_csv(filepath, index=False)
                print(f"Saved detailed action sequence for {agent_id}: {filepath}")
        
        except Exception as e:
            print(f"Error processing detailed actions for agent {agent_id}: {e}")
            continue
    
    return action_sequences


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