import numpy as np
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
def game_to_obs_vector_classic(game, agent_id, distance_map, max_distance):
    """
    Returns a vector observation for agent_id:
    For each tile type, includes:
      - distance to closest
      - distance to midpoint (between agents)
    Also includes one-hot agent inventory and other agent inventory.
    """
    tile_types = ['tomato_dispenser', 'plate_dispenser', 'cutting_board', 'delivery', 'counter']
    item_names = ['tomato', 'plate', 'tomato_cut', 'tomato_salad']

    # Get agent positions
    all_agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
    if agent_id not in all_agent_ids:
        raise ValueError(f"agent_id {agent_id} not found in gameObjects")
    other_agent_id = [aid for aid in all_agent_ids if aid != agent_id][0]
    agent = game.gameObjects[agent_id]
    other_agent = game.gameObjects[other_agent_id]
    agent_pos = (agent.slot_x, agent.slot_y)
    other_pos = (other_agent.slot_x, other_agent.slot_y)

    # Midpoint position (rounded to nearest int)
    midpoint = (int(round((agent.slot_x + other_agent.slot_x) / 2)), int(round((agent.slot_y + other_agent.slot_y) / 2)))
    obs_vector = []

    for tile_type in tile_types:
        indices = get_tile_indices_by_type(game, tile_type)
        # Only consider accessible tiles for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
        min_dist = None
        for _, x, y in accessible_agent:
            d = get_distance(distance_map, agent_pos, (x, y))
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None:
            obs_vector.append(min_dist)
        else:
            obs_vector.append(max_distance)
        # Only consider accessible tiles for midpoint
        accessible_mid = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, midpoint, (x, y)) is not None]
        min_dist_mid = None
        for _, x, y in accessible_mid:
            d = get_distance(distance_map, midpoint, (x, y))
            if min_dist_mid is None or d < min_dist_mid:
                min_dist_mid = d
        if min_dist_mid is not None:
            obs_vector.append(min_dist_mid)
        else:
            obs_vector.append(max_distance)

    # --- Add distances to items on counters ---
    for item_name in item_names:
        indices = get_item_indices_on_counters(game, item_name)
        # Only consider accessible counters for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
        min_dist = None
        for _, x, y in accessible_agent:
            d = get_distance(distance_map, agent_pos, (x, y))
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None:
            obs_vector.append(min_dist)
        else:
            obs_vector.append(max_distance)
        # Only consider accessible counters for midpoint
        accessible_mid = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, midpoint, (x, y)) is not None]
        min_dist_mid = None
        for _, x, y in accessible_mid:
            d = get_distance(distance_map, midpoint, (x, y))
            if min_dist_mid is None or d < min_dist_mid:
                min_dist_mid = d
        if min_dist_mid is not None:
            obs_vector.append(min_dist_mid)
        else:
            obs_vector.append(max_distance)

    # Distance to other agent
    dist_to_other = get_distance(distance_map, agent_pos, other_pos)
    obs_vector.append(dist_to_other if dist_to_other is not None else max_distance)

    # One-hot agent inventory
    agent_inventory = np.zeros(len(item_names), dtype=np.float32)
    item = getattr(agent, 'item', None)
    if item in item_names:
        agent_inventory[item_names.index(item)] = 1.0
    obs_vector.extend(agent_inventory.tolist())

    # One-hot other agent inventory
    other_inventory = np.zeros(len(item_names), dtype=np.float32)
    item_other = getattr(other_agent, 'item', None)
    if item_other in item_names:
        other_inventory[item_names.index(item_other)] = 1.0
    obs_vector.extend(other_inventory.tolist())
    return np.array(obs_vector, dtype=np.float32)

# ---- Competition mode with ownership awareness ---- #
def game_to_obs_vector_competition(game, agent_id, distance_map, max_distance):
    """
    Returns a vector observation for agent_id:
    For each tile type, includes:
      - distance to closest
      - distance to midpoint (between agents)
    Also includes one-hot agent inventory and other agent inventory.
    """
    tile_types = ['tomato_dispenser', 'pumpkin_dispenser', 'plate_dispenser', 'cutting_board', 'delivery', 'counter']
    item_names = ['tomato', 'pumpkin', 'plate', 'tomato_cut', 'tomato_salad']

    # Get agent positions
    all_agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
    if agent_id not in all_agent_ids:
        raise ValueError(f"agent_id {agent_id} not found in gameObjects")
    other_agent_id = [aid for aid in all_agent_ids if aid != agent_id][0]
    agent = game.gameObjects[agent_id]
    other_agent = game.gameObjects[other_agent_id]
    agent_pos = (agent.slot_x, agent.slot_y)
    other_pos = (other_agent.slot_x, other_agent.slot_y)
    midpoint = (int(round((agent.slot_x + other_agent.slot_x) / 2)), int(round((agent.slot_y + other_agent.slot_y) / 2)))
    obs_vector = []

    # --- Add distances to tile types ---
    for tile_type in tile_types:
        indices = get_tile_indices_by_type(game, tile_type)
        # Only consider accessible tiles for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
        min_dist = None
        for _, x, y in accessible_agent:
            d = get_distance(distance_map, agent_pos, (x, y))
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None:
            obs_vector.append(min_dist)
        else:
            obs_vector.append(max_distance)
        # Only consider accessible tiles for midpoint
        accessible_mid = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, midpoint, (x, y)) is not None]
        min_dist_mid = None
        for _, x, y in accessible_mid:
            d = get_distance(distance_map, midpoint, (x, y))
            if min_dist_mid is None or d < min_dist_mid:
                min_dist_mid = d
        if min_dist_mid is not None:
            obs_vector.append(min_dist_mid)
        else:
            obs_vector.append(max_distance)

    # --- Add distances to items on counters ---
    for item_name in item_names:
        indices = get_item_indices_on_counters(game, item_name)
        # Only consider accessible counters for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, agent_pos, (x, y)) is not None]
        min_dist = None
        for _, x, y in accessible_agent:
            d = get_distance(distance_map, agent_pos, (x, y))
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None:
            obs_vector.append(min_dist)
        else:
            obs_vector.append(max_distance)
        # Only consider accessible counters for midpoint
        accessible_mid = [(_idx, x, y) for (_idx, x, y) in indices if get_distance(distance_map, midpoint, (x, y)) is not None]
        min_dist_mid = None
        for _, x, y in accessible_mid:
            d = get_distance(distance_map, midpoint, (x, y))
            if min_dist_mid is None or d < min_dist_mid:
                min_dist_mid = d
        if min_dist_mid is not None:
            obs_vector.append(min_dist_mid)
        else:
            obs_vector.append(max_distance)

    # Distance to other agent
    dist_to_other = get_distance(distance_map, agent_pos, other_pos)
    obs_vector.append(dist_to_other if dist_to_other is not None else max_distance)

    # One-hot agent inventory
    agent_inventory = np.zeros(len(item_names), dtype=np.float32)
    item = getattr(agent, 'item', None)
    if item in item_names:
        agent_inventory[item_names.index(item)] = 1.0
    obs_vector.extend(agent_inventory.tolist())

    # One-hot other agent inventory
    other_inventory = np.zeros(len(item_names), dtype=np.float32)
    item_other = getattr(other_agent, 'item', None)
    if item_other in item_names:
        other_inventory[item_names.index(item_other)] = 1.0
    obs_vector.extend(other_inventory.tolist())
    return np.array(obs_vector, dtype=np.float32)

def load_max_distance(map_id, cache_dir=None):
    """
    Loads the max distance for the given map_id from cache.
    """
    if cache_dir is None:
        # Default to spoiled_broth/maps/distance_cache
        cache_dir = os.path.join(os.path.dirname(__file__), '../maps/distance_cache')
    max_dist_path = os.path.join(cache_dir, f"distance_map_{map_id}_max_distance.npy")
    if not os.path.exists(max_dist_path):
        raise FileNotFoundError(f"Max distance cache not found for map_id {map_id} at {max_dist_path}")
    return float(np.load(max_dist_path))

def normalize_obs_vector(obs_vector, max_distance):
    """
    Normalize all distance values in the observation vector by max_distance.
    Only normalize the distance values (not one-hot inventory).
    """
    obs = np.array(obs_vector, dtype=np.float32)
    # The distance values are the first N elements, rest are one-hot inventory
    # For classic: 5 tile_types * 2 + 4 item_names * 2 + 1 dist_to_other = 19 distances
    # For competition: 6 tile_types * 2 + 5 item_names * 2 + 1 dist_to_other = 27 distances
    # The rest are one-hot vectors
    if len(obs) == 27 + 10:  # competition
        dist_len = 27
    elif len(obs) == 19 + 8:  # classic
        dist_len = 19
    else:
        # fallback: try to infer
        dist_len = len(obs) - 2 * ((len(obs) - 1) // 3)
    obs[:dist_len] = obs[:dist_len] / max_distance
    return obs

# Wrapper to select observation vector function based on game_mode.
def game_to_obs_vector(game, agent_id, game_mode="classic", map_nr=None, distance_map=None, normalize=True):
    """
    Wrapper to select observation vector function based on game_mode.
    If normalize=True, normalizes the distance values using max_distance for the map.
    map_id must be provided for normalization.
    """
    max_distance = load_max_distance(map_nr)
    if game_mode == "classic":
        obs_vector = game_to_obs_vector_classic(game, agent_id, distance_map, max_distance)
    elif game_mode == "competition":
        obs_vector = game_to_obs_vector_competition(game, agent_id, distance_map, max_distance)
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")
    if normalize:
        if map_nr is None:
            raise ValueError("map_nr must be provided for normalization.")
        obs_vector = normalize_obs_vector(obs_vector, max_distance)
    return obs_vector
    