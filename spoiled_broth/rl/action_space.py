# --- New RL Action Space ---
# These are high-level, human-like actions that do not depend on the map layout

# For each action, add both 'closest' and 'midpoint' variants where relevant
RL_ACTIONS_CLASSIC = [
    "pick_up_tomato_from_dispenser",
    "pick_up_plate_from_dispenser",
    "use_cutting_board",
    "use_delivery",
    "put_down_item_on_free_counter_closest",
    "put_down_item_on_free_counter_midpoint",
    "pick_up_tomato_from_counter_closest",
    "pick_up_tomato_from_counter_midpoint",
    "pick_up_plate_from_counter_closest",
    "pick_up_plate_from_counter_midpoint",
    "pick_up_tomato_cut_from_counter_closest",
    "pick_up_tomato_cut_from_counter_midpoint",
    "pick_up_tomato_salad_from_counter_closest",
    "pick_up_tomato_salad_from_counter_midpoint"
]

RL_ACTIONS_COMPETITION = [
    "pick_up_tomato_from_dispenser",
    "pick_up_pumpkin_from_dispenser",
    "pick_up_plate_from_dispenser",
    "use_cutting_board",
    "use_delivery",
    "put_down_item_on_free_counter_closest",
    "put_down_item_on_free_counter_midpoint",
    "pick_up_tomato_from_counter_closest",
    "pick_up_tomato_from_counter_midpoint",
    "pick_up_pumpkin_from_counter_closest",
    "pick_up_pumpkin_from_counter_midpoint",
    "pick_up_plate_from_counter_closest",
    "pick_up_plate_from_counter_midpoint",
    "pick_up_tomato_cut_from_counter_closest",
    "pick_up_tomato_cut_from_counter_midpoint",
    "pick_up_pumpkin_cut_from_counter_closest",
    "pick_up_pumpkin_cut_from_counter_midpoint",
    "pick_up_tomato_salad_from_counter_closest",
    "pick_up_tomato_salad_from_counter_midpoint",
    "pick_up_pumpkin_salad_from_counter_closest",
    "pick_up_pumpkin_salad_from_counter_midpoint"
]

def get_rl_action_space(game_mode="classic"):
    if game_mode == "competition":
        return RL_ACTIONS_COMPETITION
    elif game_mode == "classic":
        return RL_ACTIONS_CLASSIC
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")

# --- Tile selection helpers using distance map ---
def find_closest_tile(agent, tile_candidates, distance_map):
    """
    agent: agent object with slot_x, slot_y
    tile_candidates: list of (idx, x, y)
    distance_map: dict[(from_x, from_y)][(to_x, to_y)] = distance
    Returns the idx of the closest tile (or None if not reachable)
    """
    agent_pos = (agent.slot_x, agent.slot_y)
    best = None
    best_dist = float('inf')
    for idx, x, y in tile_candidates:
        dist = distance_map.get(agent_pos, {}).get((x, y), None)
        if dist is not None and dist < best_dist:
            best = idx
            best_dist = dist
    return best

def find_midpoint_tile(agent, other_agent, tile_candidates, distance_map):
    """
    agent, other_agent: agent objects with slot_x, slot_y
    tile_candidates: list of (idx, x, y)
    distance_map: dict[(from_x, from_y)][(to_x, to_y)] = distance
    Returns the idx of the tile closest to the midpoint between the two agents (by path distance sum)
    """
    pos1 = (agent.slot_x, agent.slot_y)
    pos2 = (other_agent.slot_x, other_agent.slot_y)
    best = None
    best_dist = float('inf')
    for idx, x, y in tile_candidates:
        d1 = distance_map.get(pos1, {}).get((x, y), None)
        d2 = distance_map.get(pos2, {}).get((x, y), None)
        if d1 is not None and d2 is not None:
            total = d1 + d2
            if total < best_dist:
                best = idx
                best_dist = total
    return best

# Convert RL action to tile click
def convert_action_to_tile(agent, game, action_name, distance_map=None):
    """
    Given an agent, game state, and high-level action name (with _closest or _midpoint), return the tile index to click (or None for do_nothing).
    Uses the cached distance map for efficient selection.
    """
    # Parse action_name for target_mode
    if action_name.endswith("_midpoint"):
        base_action = action_name[:-9]
        target_mode = "midpoint"
    elif action_name.endswith("_closest"):
        base_action = action_name[:-8]
        target_mode = "closest"
    else:
        base_action = action_name
        target_mode = "closest"

    # Use passed distance_map, fallback to game.distance_map if not provided
    if distance_map is None:
        distance_map = getattr(game, 'distance_map', None)

    # Find the other agent (assume 2 agents)
    other_agent = None
    for a_id, a_obj in game.gameObjects.items():
        if hasattr(a_obj, 'slot_x') and a_obj is not agent:
            other_agent = a_obj
            break

    # Helper to get clickable tiles of a certain type
    def get_tiles_by_type(type_check):
        tiles = []
        grid = game.grid
        for idx in game.clickable_indices:
            x = idx % grid.width
            y = idx // grid.width
            tile = grid.tiles[x][y]
            if type_check(tile):
                tiles.append((idx, x, y))
        return tiles

    # Action logic for each base_action
    if base_action == "pick_up_tomato_from_dispenser":
        def is_tomato_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and hasattr(tile, "item") and tile.item == "tomato"
        candidates = get_tiles_by_type(is_tomato_dispenser)
    elif base_action == "pick_up_pumpkin_from_dispenser":
        def is_pumpkin_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and hasattr(tile, "item") and tile.item == "pumpkin"
        candidates = get_tiles_by_type(is_pumpkin_dispenser)
    elif base_action == "pick_up_plate_from_dispenser":
        def is_plate_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and hasattr(tile, "item") and tile.item == "plate"
        candidates = get_tiles_by_type(is_plate_dispenser)
    elif base_action == "use_cutting_board":
        def is_cutting_board(tile):
            return getattr(tile, "_type", None) == 4 and not getattr(tile, "item", None)
        candidates = get_tiles_by_type(is_cutting_board)
    elif base_action == "use_delivery":
        def is_delivery(tile):
            return getattr(tile, "_type", None) == 5
        candidates = get_tiles_by_type(is_delivery)
    elif base_action == "put_down_item_on_free_counter_closest" or base_action == "put_down_item_on_free_counter_midpoint":
        def is_free_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) is None
        candidates = get_tiles_by_type(is_free_counter)
    elif base_action == "pick_up_tomato_from_counter_closest" or base_action == "pick_up_tomato_from_counter_midpoint":
        def is_tomato_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato"
        candidates = get_tiles_by_type(is_tomato_on_counter)
    elif base_action == "pick_up_pumpkin_from_counter_closest" or base_action == "pick_up_pumpkin_from_counter_midpoint":
        def is_pumpkin_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin"
        candidates = get_tiles_by_type(is_pumpkin_on_counter)
    elif base_action == "pick_up_plate_from_counter_closest" or base_action == "pick_up_plate_from_counter_midpoint":
        def is_plate_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "plate"
        candidates = get_tiles_by_type(is_plate_on_counter)
    elif base_action == "pick_up_tomato_cut_from_counter_closest" or base_action == "pick_up_tomato_cut_from_counter_midpoint":
        def is_tomato_cut_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato_cut"
        candidates = get_tiles_by_type(is_tomato_cut_on_counter)
    elif base_action == "pick_up_pumpkin_cut_from_counter_closest" or base_action == "pick_up_pumpkin_cut_from_counter_midpoint":
        def is_pumpkin_cut_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin_cut"
        candidates = get_tiles_by_type(is_pumpkin_cut_on_counter)
    elif base_action == "pick_up_tomato_salad_from_counter_closest" or base_action == "pick_up_tomato_salad_from_counter_midpoint":
        def is_tomato_salad_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato_salad"
        candidates = get_tiles_by_type(is_tomato_salad_on_counter)
    elif base_action == "pick_up_pumpkin_salad_from_counter_closest" or base_action == "pick_up_pumpkin_salad_from_counter_midpoint":
        def is_pumpkin_salad_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin_salad"
        candidates = get_tiles_by_type(is_pumpkin_salad_on_counter)
    else:
        return None

    # Use the appropriate selection helper
    if not candidates:
        return None
    if target_mode == "closest":
        return find_closest_tile(agent, candidates, distance_map)
    elif target_mode == "midpoint" and other_agent is not None:
        return find_midpoint_tile(agent, other_agent, candidates, distance_map)
    else:
        return find_closest_tile(agent, candidates, distance_map)