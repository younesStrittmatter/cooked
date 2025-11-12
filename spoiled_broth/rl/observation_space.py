import numpy as np
import math
import os
import pickle

def get_tile_indices_by_type(game, tile_type):
    """
    Returns a list of (idx, x, y) for tiles of the given type.
    tile_type: str, one of ['tomato_dispenser', 'plate_dispenser', 'cutting_board', 'delivery', 'counter']
    """
    grid = game.grid
    indices = []
    for idx in game.clickable_indices:
        x = idx % grid.width
        y = idx // grid.width
        tile = grid.tiles[x][y]
        t = getattr(tile, '_type', None)
        item = getattr(tile, 'item', None)
        if tile_type == 'tomato_dispenser' and t == 3 and item == 'tomato':
            indices.append((idx, x, y))
        elif tile_type == 'plate_dispenser' and t == 3 and item == 'plate':
            indices.append((idx, x, y))
        elif tile_type == 'cutting_board' and t == 4:
            indices.append((idx, x, y))
        elif tile_type == 'delivery' and t == 5:
            indices.append((idx, x, y))
        elif tile_type == 'counter' and t == 2:
            indices.append((idx, x, y))
    return indices

def get_item_indices_on_counters(game, item_name):
    """
    Returns a list of (idx, x, y) for counters with the given item_name.
    """
    grid = game.grid
    indices = []
    for idx in game.clickable_indices:
        x = idx % grid.width
        y = idx // grid.width
        tile = grid.tiles[x][y]
        t = getattr(tile, '_type', None)
        item = getattr(tile, 'item', None)
        if t == 2 and item == item_name:
            indices.append((idx, x, y))
    return indices

def get_distance(distance_map, from_xy, to_xy):
    if distance_map is None:
        raise ValueError("distance_map should never be None. It must be provided and loaded before calling get_distance.")
    # Support both original nested-dict and DistanceMatrixWrapper-like objects
    try:
        if isinstance(distance_map, dict):
            return distance_map.get(from_xy, {}).get(to_xy, None)
        # Otherwise assume it has a .get(from_xy) -> dict-like
        inner = distance_map.get(from_xy)
        if inner is None:
            return None
        return inner.get(to_xy, None)
    except Exception:
        return None

# ---- Classic mode without ownership awareness ---- #
def game_to_obs_vector_classic(game, agent_id, distance_map):
    """
    Returns a vector observation for agent_id:
    For each tile type, includes:
      - distance to closest
      - distance to midpoint (between agents)
    Also includes one-hot agent inventory and other agent inventory.
    """
    normalization_factor = game.normalization_factor

    tile_types = ['tomato_dispenser', 'plate_dispenser', 'cutting_board', 'delivery']
    item_names = [None, 'tomato', 'plate', 'tomato_cut', 'tomato_salad']

    # Get agent walking and cutting speeds
    agent_walking_speed = game.walking_speeds.get(agent_id, 1) * game.walking_time
    agent_cutting_speed = game.cutting_speeds.get(agent_id, 1) * game.cutting_time 

    # Get agent positions
    all_agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
    if agent_id not in all_agent_ids:
        raise ValueError(f"agent_id {agent_id} not found in gameObjects")
    
    # Handle single agent case
    other_agent_ids = [aid for aid in all_agent_ids if aid != agent_id]
    has_other_agent = len(other_agent_ids) > 0
    
    agent = game.gameObjects[agent_id]
    agent_pos = (agent.slot_x, agent.slot_y)
    
    if has_other_agent:
        other_agent_id = other_agent_ids[0]
        other_agent = game.gameObjects[other_agent_id]
        other_pos = (other_agent.slot_x, other_agent.slot_y)
        # Midpoint position (rounded to nearest int)
        midpoint = (int(round((agent.slot_x + other_agent.slot_x) / 2)), int(round((agent.slot_y + other_agent.slot_y) / 2)))
    else:
        other_agent = None
        other_pos = None  # No other agent
        # For single agent, calculate midpoint as closest accessible tile position
        all_tiles = []
        for tile_type in tile_types:
            indices = get_tile_indices_by_type(game, tile_type)
            accessible = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
            all_tiles.extend(accessible)
        
        if all_tiles:
            # Find the closest tile to agent
            closest_tile = min(all_tiles, key=lambda t: get_distance(distance_map, agent_pos, (t[1], t[2])))
            midpoint = (closest_tile[1], closest_tile[2])
        else:
            # Fallback to agent position if no accessible tiles
            midpoint = (agent.slot_x, agent.slot_y)
    obs_vector = []

    # --- Add times to tile types ---
    for tile_type in tile_types:
        if tile_type == 'cutting_board':
            action_time = agent_cutting_speed 
        else:
            action_time = 0

        indices = get_tile_indices_by_type(game, tile_type)

        # Only consider accessible tiles for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
        min_dist = None
        for _, x, y in accessible_agent:
            d = get_distance(distance_map, agent_pos, (x, y))
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None:
            time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
            obs_vector.append(time_to_tile)
        else:
            obs_vector.append(1)


    # --- Add times to items on counters ---
    for item_name in item_names:
        action_time = 0
        indices = get_item_indices_on_counters(game, item_name)

        if len(indices) > 0:
            obs_vector.append(1) # There is a tile with that item
            # Only consider accessible counters for agent
            accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
            min_dist = None
            for _, x, y in accessible_agent:
                d = get_distance(distance_map, agent_pos, (x, y))
                if min_dist is None or d < min_dist:
                    min_dist = d
            if min_dist is not None:
                time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
                obs_vector.append(time_to_tile)
            else:
                obs_vector.append(1)

            # Choose the counter tile that is closest to the midpoint using
            # Euclidean distance (this ignores path accessibility from the midpoint).
            # Then compute the agent->tile path distance using get_distance so
            # the agent time accounts for obstacles; if agent cannot reach the
            # chosen tile, fall back to max_distance.
            # pick tile index with minimum Euclidean distance to midpoint
            best = min(indices, key=lambda it: math.hypot(it[1] - midpoint[0], it[2] - midpoint[1]))
            _, bx, by = best
            path_dist = get_distance(distance_map, agent_pos, (bx, by))
            if path_dist is not None:
                time_to_midtile = (path_dist / agent_walking_speed + action_time) / normalization_factor
            else:
                time_to_midtile = 1
            obs_vector.append(time_to_midtile)
        else:
            # no counters with that item: append fallbacks for agent time and midpoint time
            obs_vector.append(0) # There are no tiles with that item
            obs_vector.append(1) # Fallback for agent time
            obs_vector.append(1) # Fallback for midpoint time

    # Distance to other agent - both Euclidean and pathfinding distances
    # For single agent case, use default values
    if has_other_agent:
        # 1. Euclidean distance (straight-line distance)
        euclidean_dist = math.sqrt((agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2)
        euclidean_time = (euclidean_dist / agent_walking_speed) / normalization_factor
        obs_vector.append(euclidean_time)
        
        # 2. Pathfinding distance (using distance map for accessibility)
        pathfinding_dist = get_distance(distance_map, agent_pos, other_pos)
        pathfinding_time = (pathfinding_dist / agent_walking_speed) / normalization_factor if pathfinding_dist is not None else 1
        obs_vector.append(pathfinding_time)
    else:
        # Single agent case: append default values for other agent distances
        obs_vector.append(1.0)  # No other agent, use max distance
        obs_vector.append(1.0)  # No other agent, use max distance

    # One-hot agent inventory
    agent_inventory = np.zeros(len(item_names), dtype=np.float32)
    item = getattr(agent, 'item', None)
    if item in item_names:
        agent_inventory[item_names.index(item)] = 1.0
    obs_vector.extend(agent_inventory.tolist())

    # One-hot other agent inventory
    if has_other_agent:
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        item_other = getattr(other_agent, 'item', None)
        if item_other in item_names:
            other_inventory[item_names.index(item_other)] = 1.0
        obs_vector.extend(other_inventory.tolist())
    else:
        # Single agent case: append zeros for other agent inventory
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        obs_vector.extend(other_inventory.tolist())
    return np.array(obs_vector, dtype=np.float32)

# ---- Competition mode with ownership awareness ---- #
def game_to_obs_vector_competition(game, agent_id, distance_map):
    """
    Returns a vector observation for agent_id:
    For each tile type, includes:
      - distance to closest
      - distance to midpoint (between agents)
    Also includes one-hot agent inventory and other agent inventory.
    """
    normalization_factor = game.normalization_factor

    tile_types = ['tomato_dispenser', 'pumpkin_dispenser', 'plate_dispenser', 'cutting_board', 'delivery']
    item_names = [None, 'tomato', 'pumpkin', 'plate', 'tomato_cut', 'pumpkin_cut', 'tomato_salad', 'pumpkin_salad']

    # Get agent walking and cutting speeds
    agent_walking_speed = game.walking_speeds.get(agent_id, 1)
    agent_cutting_speed = game.cutting_speeds.get(agent_id, 1)

    # Get agent positions
    all_agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
    if agent_id not in all_agent_ids:
        raise ValueError(f"agent_id {agent_id} not found in gameObjects")
    
    # Handle single agent case
    other_agent_ids = [aid for aid in all_agent_ids if aid != agent_id]
    has_other_agent = len(other_agent_ids) > 0
    
    agent = game.gameObjects[agent_id]
    agent_pos = (agent.slot_x, agent.slot_y)
    
    if has_other_agent:
        other_agent_id = other_agent_ids[0]
        other_agent = game.gameObjects[other_agent_id]
        other_pos = (other_agent.slot_x, other_agent.slot_y)
        midpoint = (int(round((agent.slot_x + other_agent.slot_x) / 2)), int(round((agent.slot_y + other_agent.slot_y) / 2)))
    else:
        other_agent = None
        other_pos = None  # No other agent
        # For single agent, calculate midpoint as closest accessible tile position
        all_tiles = []
        for tile_type in tile_types:
            indices = get_tile_indices_by_type(game, tile_type)
            accessible = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
            all_tiles.extend(accessible)
        
        if all_tiles:
            # Find the closest tile to agent
            closest_tile = min(all_tiles, key=lambda t: get_distance(distance_map, agent_pos, (t[1], t[2])))
            midpoint = (closest_tile[1], closest_tile[2])
        else:
            # Fallback to agent position if no accessible tiles
            midpoint = (agent.slot_x, agent.slot_y)
    obs_vector = []

    # --- Add distances to tile types ---
    for tile_type in tile_types:
        if tile_type == 'cutting_board':
            action_time = agent_cutting_speed
        else:
            action_time = 0

        indices = get_tile_indices_by_type(game, tile_type)
        # Only consider accessible tiles for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
        min_dist = None
        for _, x, y in accessible_agent:
            d = get_distance(distance_map, agent_pos, (x, y))
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None:
            time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
            obs_vector.append(time_to_tile)
        else:
            obs_vector.append(1)

    # --- Add distances to items on counters ---
    for item_name in item_names:
        action_time = 0
        indices = get_item_indices_on_counters(game, item_name)

        if len(indices) > 0:
            obs_vector.append(1) # There is a tile with that item
            # Only consider accessible counters for agent
            accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
            min_dist = None
            for _, x, y in accessible_agent:
                d = get_distance(distance_map, agent_pos, (x, y))
                if min_dist is None or d < min_dist:
                    min_dist = d
            if min_dist is not None:
                time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
                obs_vector.append(time_to_tile)
            else:
                obs_vector.append(1)

            # Choose the counter tile that is closest to the midpoint by Euclidean
            # distance (ignore midpoint accessibility). For the agent->tile value
            # use the path distance from the agent to that chosen tile (fall back to
            # max_distance if unreachable). For competition mode we keep the original
            # convention of appending agent path distance then midpoint Euclidean distance.
            best = min(indices, key=lambda it: math.hypot(it[1] - midpoint[0], it[2] - midpoint[1]))
            _, bx, by = best
            path_dist = get_distance(distance_map, agent_pos, (bx, by))
            if path_dist is not None:
                time_to_midtile = (path_dist / agent_walking_speed + action_time) / normalization_factor
                obs_vector.append(time_to_midtile)
            else:
                obs_vector.append(1)
        else:
            # no counters with that item: append fallbacks for agent distance and midpoint distance
            obs_vector.append(0) # There are no tiles with that item
            obs_vector.append(1) # Fallback for agent time
            obs_vector.append(1) # Fallback for midpoint time

    # Distance to other agent - both Euclidean and pathfinding distances
    if has_other_agent:
        # 1. Euclidean distance (straight-line distance)
        euclidean_dist = math.sqrt((agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2)
        euclidean_time = (euclidean_dist / agent_walking_speed) / normalization_factor
        obs_vector.append(euclidean_time)
        
        # 2. Pathfinding distance (using distance map for accessibility)
        pathfinding_dist = get_distance(distance_map, agent_pos, other_pos)
        pathfinding_time = (pathfinding_dist / agent_walking_speed) / normalization_factor if pathfinding_dist is not None else 1
        obs_vector.append(pathfinding_time)
    else:
        # Single agent case: always use 1.0 (normalized max distance) for other agent distances
        obs_vector.append(1.0)
        obs_vector.append(1.0)

    # One-hot agent inventory
    agent_inventory = np.zeros(len(item_names), dtype=np.float32)
    item = getattr(agent, 'item', None)
    if item in item_names:
        agent_inventory[item_names.index(item)] = 1.0
    obs_vector.extend(agent_inventory.tolist())

    # One-hot other agent inventory
    if has_other_agent:
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        item_other = getattr(other_agent, 'item', None)
        if item_other in item_names:
            other_inventory[item_names.index(item_other)] = 1.0
        obs_vector.extend(other_inventory.tolist())
    else:
        # Single agent case: append zeros for other agent inventory
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        obs_vector.extend(other_inventory.tolist())
    return np.array(obs_vector, dtype=np.float32)

# Wrapper to select observation vector function based on game_mode.
def game_to_obs_vector(game, agent_id, game_mode="classic", distance_map=None):
    """
    Wrapper to select observation vector function based on game_mode.
    If normalize=True, normalizes the distance values using max_distance for the map.
    map_id must be provided for normalization.
    """
    if game_mode == "classic":
        obs_vector = game_to_obs_vector_classic(game, agent_id, distance_map)
    elif game_mode == "competition":
        obs_vector = game_to_obs_vector_competition(game, agent_id, distance_map)
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")
    return obs_vector
    