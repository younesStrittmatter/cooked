ACTION_TYPE_DO_NOTHING = "do_nothing"
ACTION_TYPE_FLOOR = "floor"
ACTION_TYPE_WALL = "wall"
ACTION_TYPE_USELESS_COUNTER = "useless_counter"
ACTION_TYPE_USEFUL_COUNTER = "useful_counter"
ACTION_TYPE_USEFUL_FOOD_DISPENSER = "useful_food_dispenser"
ACTION_TYPE_USELESS_FOOD_DISPENSER = "useless_food_dispenser"
ACTION_TYPE_USELESS_CUTTING_BOARD = "useless_cutting_board"
ACTION_TYPE_USEFUL_CUTTING_BOARD = "useful_cutting_board"
ACTION_TYPE_USEFUL_PLATE_DISPENSER = "useful_plate_dispenser"
ACTION_TYPE_USELESS_PLATE_DISPENSER = "useless_plate_dispenser"
ACTION_TYPE_USELESS_DELIVERY = "useless_delivery"
ACTION_TYPE_USEFUL_DELIVERY = "useful_delivery"
ACTION_TYPE_INACCESSIBLE = "inaccessible_tile"

# Detailed action types considering ownership and usefulness (competition mode)
ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_own"
ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER = "useless_food_dispenser_own"
ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_other"
ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER = "useless_food_dispenser_other"
ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD = "useful_cutting_board_own"
ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD = "useful_cutting_board_other"
ACTION_TYPE_OWN_USEFUL_DELIVERY = "useful_delivery_own"
ACTION_TYPE_OTHER_USEFUL_DELIVERY = "useful_delivery_other"

# Centralized list of all action types for import
ACTION_TYPE_LIST_CLASSIC = [
    ACTION_TYPE_DO_NOTHING,
    ACTION_TYPE_FLOOR,
    ACTION_TYPE_WALL,
    ACTION_TYPE_USELESS_COUNTER,
    ACTION_TYPE_USEFUL_COUNTER,
    ACTION_TYPE_USELESS_FOOD_DISPENSER,
    ACTION_TYPE_USEFUL_FOOD_DISPENSER,
    ACTION_TYPE_USELESS_CUTTING_BOARD,
    ACTION_TYPE_USEFUL_CUTTING_BOARD,
    ACTION_TYPE_USELESS_PLATE_DISPENSER,
    ACTION_TYPE_USEFUL_PLATE_DISPENSER,
    ACTION_TYPE_USELESS_DELIVERY,
    ACTION_TYPE_USEFUL_DELIVERY,
    ACTION_TYPE_INACCESSIBLE
]

ACTION_TYPE_LIST_COMPETITION = [
    ACTION_TYPE_DO_NOTHING,
    ACTION_TYPE_FLOOR,
    ACTION_TYPE_WALL,
    ACTION_TYPE_USELESS_COUNTER,
    ACTION_TYPE_USEFUL_COUNTER,
    ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER,
    ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER,
    ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER,
    ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER,
    ACTION_TYPE_USELESS_CUTTING_BOARD,
    ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD,
    ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD,
    ACTION_TYPE_USELESS_PLATE_DISPENSER,
    ACTION_TYPE_USEFUL_PLATE_DISPENSER,
    ACTION_TYPE_USELESS_DELIVERY,
    ACTION_TYPE_OWN_USEFUL_DELIVERY,
    ACTION_TYPE_OTHER_USEFUL_DELIVERY,
    ACTION_TYPE_INACCESSIBLE
]

# Helper: wrapper functions to get action type based on game mode
def get_action_type_list(game_mode):
    """
    Returns the list of action types based on the game mode.
    """
    if game_mode == "classic":
        return ACTION_TYPE_LIST_CLASSIC
    elif game_mode == "competition":
        return ACTION_TYPE_LIST_COMPETITION
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")

# ---- Classic mode without ownership awareness ---- #
def get_action_type_classic(tile, agent, x=None, y=None, accessibility_map=None):
    """
    Determine the type of action based on the tile clicked and agent state.
    """
    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_FLOOR  # Default to floor for None/invalid tiles
        
    # Check accessibility using pre-computed map
    agent_pos = (agent.slot_x, agent.slot_y)
    tile_pos = (x, y)
    if accessibility_map is not None and tile_pos not in accessibility_map.get(agent_pos, set()):
        return ACTION_TYPE_INACCESSIBLE

    # Default mapping for unknown tiles
    default_action = ACTION_TYPE_FLOOR

    # Agent holds something? (e.g., tomato, salad, etc.)
    holding_something = getattr(agent, "item", None) is not None

    # Tile type 0: floor
    if tile._type == 0:
        return ACTION_TYPE_FLOOR

    # Tile type 1: wall
    if tile._type == 1:
        return ACTION_TYPE_WALL

    # Tile type 2: counter
    if tile._type == 2:
        # Useful if agent is holding something (can drop it), or counter has something and agent can pick it
        has_on_counter = getattr(tile, "item", None) is not None
        if holding_something or has_on_counter:
            return ACTION_TYPE_USEFUL_COUNTER
        else:
            return ACTION_TYPE_USELESS_COUNTER

    # Tile type 3: dispenser
    if tile._type == 3:
        if getattr(tile, "item", None) == "plate":
            return ACTION_TYPE_USEFUL_PLATE_DISPENSER if not holding_something else ACTION_TYPE_USELESS_PLATE_DISPENSER
        else:
            return ACTION_TYPE_USEFUL_FOOD_DISPENSER if not holding_something else ACTION_TYPE_USELESS_FOOD_DISPENSER
            
    # Tile type 4: cutting board
    if tile._type == 4:
        # Check if cutting board has an item
        board_item = getattr(tile, "item", None)

        # Agent holds an item that can be cut
        item_in_hand = getattr(agent, "item", None)
        holding_uncut = item_in_hand is not None and getattr(item_in_hand, "cut_stage", 0) == 0
        
        if holding_uncut or board_item is not None:
            return ACTION_TYPE_USEFUL_CUTTING_BOARD
        else:
            return ACTION_TYPE_USELESS_CUTTING_BOARD

    # Tile type 5: delivery
    if tile._type == 5:
        # Useful if agent holds a deliverable item (e.g., salad)
        item = getattr(agent, "item", None)
        valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']

        if item is not None and item in valid_items:
            return ACTION_TYPE_USEFUL_DELIVERY
        else:
            return ACTION_TYPE_USELESS_DELIVERY

    # Default fallback
    return default_action

# ---- Competition mode with ownership awareness ---- #
def get_action_type_competition(tile, agent, agent_id, agent_food_type):
    """
    Determine the type of action based on the tile clicked and agent state.
    Uses agent_food_type to distinguish own food from other food.
    """
    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_FLOOR  # Default to floor for None/invalid tiles

    default_action = ACTION_TYPE_FLOOR
    holding_item = getattr(agent, "item", None)
    holding_something = holding_item is not None

    own_food = agent_food_type.get(agent_id, None)
    other_food = None

    for aid, food in agent_food_type.items():
        if aid != agent_id:
            other_food = food

    # Tile type 0: floor
    if tile._type == 0:
        return ACTION_TYPE_FLOOR

    # Tile type 1: wall
    if tile._type == 1:
        return ACTION_TYPE_WALL

    # Tile type 2: counter
    if tile._type == 2:
        # Useful if agent is holding something (can drop it), or counter has something and agent can pick it
        item_on_counter = getattr(tile, "item", None)
        if holding_something or item_on_counter is not None:
            return ACTION_TYPE_USEFUL_COUNTER
        else:
            return ACTION_TYPE_USELESS_COUNTER

    # Tile type 3: dispenser
    if tile._type == 3:
        disp_item = getattr(tile, "item", None)
        if disp_item == "plate":
            return ACTION_TYPE_USEFUL_PLATE_DISPENSER if not holding_something else ACTION_TYPE_USELESS_PLATE_DISPENSER
        elif disp_item == own_food:
            return ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER if not holding_something else ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER
        elif disp_item == other_food:
            return ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER if not holding_something else ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER
            
    # Tile type 4: cutting board
    if tile._type == 4:
        board_item = getattr(tile, "item", None)

        # Detect food type on board
        board_item_type = board_item if isinstance(board_item, str) else None

        # Item in hand can be cut (not already cut/salad)
        item_in_hand = holding_item
        holding_uncut = (
            item_in_hand is not None and
            isinstance(item_in_hand, str) and
            not item_in_hand.endswith("_cut") and
            not item_in_hand.endswith("_salad")
        )

        # Check if it's own or other food
        if holding_uncut:
            if item_in_hand == own_food:
                return ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD
            elif item_in_hand == other_food:
                return ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD
        elif board_item_type:
            if board_item_type.startswith(own_food):
                return ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD
            elif board_item_type.startswith(other_food):
                return ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD

        return ACTION_TYPE_USELESS_CUTTING_BOARD

    # Tile type 5: delivery
    if tile._type == 5:
        item = holding_item
        if item is None or not isinstance(item, str):
            return ACTION_TYPE_USELESS_DELIVERY

        # Detect if item is deliverable
        deliverable = item in ['tomato_salad', 'pumpkin_salad']

        if deliverable:
            if item.startswith(own_food):
                return ACTION_TYPE_OWN_USEFUL_DELIVERY
            elif item.startswith(other_food):
                return ACTION_TYPE_OTHER_USEFUL_DELIVERY
        else:
            return ACTION_TYPE_USELESS_DELIVERY

    # Default fallback
    return default_action

def get_action_type(tile, agent, agent_id=None, agent_food_type=None, game_mode="classic", x=None, y=None, accessibility_map=None):
    """
    Wrapper to select appropriate action type function based on game mode.
    """
    if game_mode == "classic":
        return get_action_type_classic(tile, agent, x=x, y=y, accessibility_map=accessibility_map)
    elif game_mode == "competition":
        return get_action_type_competition(tile, agent, agent_id, agent_food_type)
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")