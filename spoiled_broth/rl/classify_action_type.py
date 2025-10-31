ACTION_TYPE_DO_NOTHING = "do_nothing"
ACTION_TYPE_FLOOR = "useless_floor"
ACTION_TYPE_WALL = "useless_wall"
ACTION_TYPE_USELESS_COUNTER = "useless_counter"
ACTION_TYPE_USEFUL_COUNTER = "useful_counter"
ACTION_TYPE_SALAD_ASSEMBLY = "salad_assembly"
ACTION_TYPE_USEFUL_FOOD_DISPENSER = "useful_food_dispenser"
ACTION_TYPE_USELESS_FOOD_DISPENSER = "destructive_food_dispenser"
ACTION_TYPE_USELESS_CUTTING_BOARD = "useless_cutting_board"
ACTION_TYPE_USEFUL_CUTTING_BOARD = "useful_cutting_board"
ACTION_TYPE_USEFUL_PLATE_DISPENSER = "useful_plate_dispenser"
ACTION_TYPE_USELESS_PLATE_DISPENSER = "destructive_plate_dispenser"
ACTION_TYPE_USELESS_DELIVERY = "useless_delivery"
ACTION_TYPE_USEFUL_DELIVERY = "useful_delivery"
ACTION_TYPE_INACCESSIBLE = "inaccessible_tile"

# Detailed action types considering ownership and usefulness (competition mode)
ACTION_TYPE_OWN_SALAD_ASSEMBLY = "salad_assembly_own"
ACTION_TYPE_OTHER_SALAD_ASSEMBLY = "salad_assembly_other"
ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_own"
ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER = "destructive_food_dispenser_own"
ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_other"
ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER = "destructive_food_dispenser_other"
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
    ACTION_TYPE_SALAD_ASSEMBLY,
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
    ACTION_TYPE_OWN_SALAD_ASSEMBLY,
    ACTION_TYPE_OTHER_SALAD_ASSEMBLY,
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
def get_action_type_classic(self, tile, agent, agent_id, agent_events, x, y, accessibility_map):
    """
    Determine the type of action based on the tile clicked and agent state.
    """

    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_FLOOR, agent_events  # Default to floor for None/invalid tiles

    # Check accessibility using pre-computed map
    agent_pos = (agent.slot_x, agent.slot_y)
    tile_pos = (x, y)
    if accessibility_map is not None and tile_pos not in accessibility_map.get(agent_pos, set()):
        return ACTION_TYPE_INACCESSIBLE, agent_events

    # Agent holds something? (e.g., tomato, salad, etc.)
    holding_item = getattr(agent, "item", None)

    # Tile type 0: floor
    if tile._type == 0:
        return ACTION_TYPE_FLOOR, agent_events

    # Tile type 1: wall
    if tile._type == 1:
        return ACTION_TYPE_WALL, agent_events

    # Tile type 2: counter
    if tile._type == 2:
        # Useful if agent is holding something (can drop it), or counter has something and agent can pick it
        item_on_counter = getattr(tile, "item", None)
        valid_items = ["tomato_cut", "pumpkin_cut", "cabbage_cut"]
        if (holding_item == "plate" and item_on_counter in valid_items) or (holding_item in valid_items and item_on_counter == "plate"):
            agent_events[agent_id]["salad"] += 1
            self.total_agent_events[agent_id]["salad"] += 1
            return ACTION_TYPE_SALAD_ASSEMBLY, agent_events
        elif holding_item is not None or item_on_counter is not None:
            agent_events[agent_id]["counter"] += 1
            self.total_agent_events[agent_id]["counter"] += 1
            return ACTION_TYPE_USEFUL_COUNTER, agent_events
        else:
            return ACTION_TYPE_USELESS_COUNTER, agent_events

    # Tile type 3: dispenser
    if tile._type == 3:
        if getattr(tile, "item", None) == "plate":
            if holding_item is None:
                agent_events[agent_id]["plate"] += 1
                self.total_agent_events[agent_id]["plate"] += 1
                return ACTION_TYPE_USEFUL_PLATE_DISPENSER, agent_events
            else:
                return ACTION_TYPE_USELESS_PLATE_DISPENSER, agent_events
        else:
            if holding_item is None:
                agent_events[agent_id]["raw_food"] += 1
                self.total_agent_events[agent_id]["raw_food"] += 1
                return ACTION_TYPE_USEFUL_FOOD_DISPENSER, agent_events
            else:
                return ACTION_TYPE_USELESS_FOOD_DISPENSER, agent_events
            
    # Tile type 4: cutting board
    if tile._type == 4:
        # Agent holds an item that can be cut
        valid_items = ["tomato", "pumpkin", "cabbage"]
        if holding_item in valid_items and self.counted_cuts[agent_id] is False:
            # Only count cut if not already counted for this item
            agent_events[agent_id]["cut"] += 1
            self.total_agent_events[agent_id]["cut"] += 1
            self.counted_cuts[agent_id] = True
            return ACTION_TYPE_USEFUL_CUTTING_BOARD, agent_events
        else:
            return ACTION_TYPE_USELESS_CUTTING_BOARD, agent_events

    # Tile type 5: delivery
    if tile._type == 5:
        # Useful if agent holds a deliverable item (e.g., salad)
        valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']

        if holding_item in valid_items:
            agent_events[agent_id]["deliver"] += 1
            self.total_agent_events[agent_id]["deliver"] += 1
            return ACTION_TYPE_USEFUL_DELIVERY, agent_events
        else:
            return ACTION_TYPE_USELESS_DELIVERY, agent_events

    # Default fallback
    return ACTION_TYPE_FLOOR, agent_events

# ---- Competition mode with ownership awareness ---- #
def get_action_type_competition(self, tile, agent, agent_id, agent_events, agent_food_type, x, y, accessibility_map):
    """
    Determine the type of action based on the tile clicked and agent state.
    Uses agent_food_type to distinguish own food from other food.
    """

    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_FLOOR, agent_events  # Default to floor for None/invalid tiles
    
    # Check accessibility using pre-computed map
    agent_pos = (agent.slot_x, agent.slot_y)
    tile_pos = (x, y)
    if accessibility_map is not None and tile_pos not in accessibility_map.get(agent_pos, set()):
        return ACTION_TYPE_INACCESSIBLE, agent_events

    holding_item = getattr(agent, "item", None)

    own_food = agent_food_type.get(agent_id, None)
    other_food = None

    for aid, food in agent_food_type.items():
        if aid != agent_id:
            other_food = food

    # Tile type 0: floor
    if tile._type == 0:
        return ACTION_TYPE_FLOOR, agent_events

    # Tile type 1: wall
    if tile._type == 1:
        return ACTION_TYPE_WALL, agent_events

    # Tile type 2: counter
    if tile._type == 2:
        # Useful if agent is holding something (can drop it), or counter has something and agent can pick it
        item_on_counter = getattr(tile, "item", None)
        valid_items = ["tomato_cut", "pumpkin_cut", "cabbage_cut"]
        if (holding_item in valid_items and item_on_counter == "plate") or (holding_item == "plate" and item_on_counter in valid_items):
            if holding_item in valid_items:
                if holding_item.startswith(own_food):
                    agent_events[agent_id]["salad_own"] += 1
                    self.total_agent_events[agent_id]["salad_own"] += 1
                    return ACTION_TYPE_OWN_SALAD_ASSEMBLY, agent_events
                elif holding_item.startswith(other_food):
                    agent_events[agent_id]["salad_other"] += 1
                    self.total_agent_events[agent_id]["salad_other"] += 1
                    return ACTION_TYPE_OTHER_SALAD_ASSEMBLY, agent_events
            else:
                if item_on_counter.startswith(own_food):
                    agent_events[agent_id]["salad_own"] += 1
                    self.total_agent_events[agent_id]["salad_own"] += 1
                    return ACTION_TYPE_OWN_SALAD_ASSEMBLY, agent_events
                elif item_on_counter.startswith(other_food):
                    agent_events[agent_id]["salad_other"] += 1
                    self.total_agent_events[agent_id]["salad_other"] += 1
                    return ACTION_TYPE_OTHER_SALAD_ASSEMBLY, agent_events
        elif holding_item is not None or item_on_counter is not None:
            agent_events[agent_id]["counter"] += 1
            self.total_agent_events[agent_id]["counter"] += 1
            return ACTION_TYPE_USEFUL_COUNTER, agent_events
        else:
            return ACTION_TYPE_USELESS_COUNTER, agent_events

    # Tile type 3: dispenser
    if tile._type == 3:
        disp_item = getattr(tile, "item", None)
        if disp_item == "plate":
            if holding_item is None:
                agent_events[agent_id]["plate"] += 1
                self.total_agent_events[agent_id]["plate"] += 1
                return ACTION_TYPE_USEFUL_PLATE_DISPENSER, agent_events
            else:
                return ACTION_TYPE_USELESS_PLATE_DISPENSER, agent_events
        elif disp_item == own_food:
            if holding_item is None:
                agent_events[agent_id]["raw_food_own"] += 1
                self.total_agent_events[agent_id]["raw_food_own"] += 1
                return ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER, agent_events
            else:
                return ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER, agent_events
        elif disp_item == other_food:
            if holding_item is None:
                agent_events[agent_id]["raw_food_other"] += 1
                self.total_agent_events[agent_id]["raw_food_other"] += 1
                return ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER, agent_events
            else:
                return ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER, agent_events
            
    # Tile type 4: cutting board
    if tile._type == 4:
        # Item in hand can be cut (not already cut/salad)
        valid_items = ["tomato", "pumpkin", "cabbage"]
        # Check if it's own or other food
        if holding_item in valid_items and self.counted_cuts.get(agent_id, False) is False:
            if holding_item == own_food:
                agent_events[agent_id]["cut_own"] += 1
                self.total_agent_events[agent_id]["cut_own"] += 1
                self.counted_cuts[agent_id] = True
                return ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD, agent_events
            elif holding_item == other_food:
                agent_events[agent_id]["cut_other"] += 1
                self.total_agent_events[agent_id]["cut_other"] += 1
            return ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD, agent_events
        return ACTION_TYPE_USELESS_CUTTING_BOARD, agent_events

    # Tile type 5: delivery
    if tile._type == 5:
        # Useful if agent holds a deliverable item (e.g., salad)
        valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']

        if holding_item in valid_items:
            if holding_item.startswith(own_food):
                agent_events[agent_id]["deliver_own"] += 1
                self.total_agent_events[agent_id]["deliver_own"] += 1
                return ACTION_TYPE_OWN_USEFUL_DELIVERY, agent_events
            elif holding_item.startswith(other_food):
                agent_events[agent_id]["deliver_other"] += 1
                self.total_agent_events[agent_id]["deliver_other"] += 1
                return ACTION_TYPE_OTHER_USEFUL_DELIVERY, agent_events
        else:
            return ACTION_TYPE_USELESS_DELIVERY, agent_events

    # Default fallback
    return ACTION_TYPE_FLOOR, agent_events

def get_action_type_and_agent_events(self, tile, agent, agent_id=None, agent_events=None, agent_food_type=None, game_mode="classic", x=None, y=None, accessibility_map=None):
    """
    Wrapper to select appropriate action type function based on game mode.
    """
    if game_mode == "classic":
        return get_action_type_classic(self, tile, agent, agent_id, agent_events, x=x, y=y, accessibility_map=accessibility_map)
    elif game_mode == "competition":
        return get_action_type_competition(self, tile, agent, agent_id, agent_events, agent_food_type, x=x, y=y, accessibility_map=accessibility_map)
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")