"""
Optimized game step functions for CPU execution.
These functions handle game state updates efficiently on CPU while
neural network training happens on GPU in parallel.
"""
from engine.extensions.topDownGridWorld.a_star import grid_to_world, Node, find_path, path_length
from spoiled_broth.rl.action_space import get_rl_action_space
import math

def update_agents_directly(self, validated_actions, advanced_time, action_info, agent_events):
    """
    Directly update game state without going through the engine.
    Handles agent movement, action completion, and state updates.
    """
    
    # Process each agent
    for agent_id, agent in self.agent_map.items():
        # Handle movement if agent has a path
        if hasattr(agent, 'path') and agent.path and hasattr(agent, 'path_index'):
            update_agent_movement(self, agent, advanced_time)
        
        # Handle action completion if agent was performing an action
        if agent_id in action_info and hasattr(agent, 'busy_until'):
            action_data = action_info[agent_id]
            # Check if action should complete within advanced_time
            if agent.busy_until <= self._elapsed_time:
                complete_agent_action(self, agent_id, agent, action_data, agent_events)

def update_agent_movement(self, agent, delta_time):
    """Update agent position along their path using A* trajectory and walking speed."""
    if not hasattr(agent, 'path') or not agent.path or not hasattr(agent, 'path_index'):
        return
    
    PHYSICS_STEPS = 100
    TILE_SNAP_THRESHOLD = 0.2
    
    def close(a, b, tolerance=TILE_SNAP_THRESHOLD):
        return abs(a - b) < tolerance
    
    d_t = delta_time / PHYSICS_STEPS
    
    for i in range(PHYSICS_STEPS):
        if agent.path_index >= len(agent.path):
            return  # Finished path
            
        next_tile = agent.path[agent.path_index]
        target_pos = grid_to_world(next_tile, self.game.grid)
        dx = target_pos['x'] - agent.x
        dy = target_pos['y'] - agent.y
        
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        if magnitude > 0:
            # Use agent's walking speed
            speed = getattr(agent, 'speed', 30)  # Default speed if not set
            agent.x += dx / magnitude * speed * d_t
            agent.y += dy / magnitude * speed * d_t
        
        # Check if close enough to snap to tile
        if close(agent.x, target_pos['x']) and close(agent.y, target_pos['y']):
            agent.path_index += 1
            # Note: slot_x and slot_y are calculated automatically from x and y pixel coordinates

def complete_agent_action(self, agent_id, agent, action_data, agent_events):
    """Complete an agent's action by updating game state directly."""
    tile_index = action_data['tile_index']
    action_type = action_data['action_type']
    
    # Get the tile
    grid_w = self.game.grid.width
    x = tile_index % grid_w
    y = tile_index // grid_w
    tile = self.game.grid.tiles[x][y]
    
    # Ensure agent has finished moving to the target (if it's not walkable, position next to it)
    if hasattr(tile, 'is_walkable') and tile.is_walkable:
        agent.x = tile.slot_x * self.game.grid.tile_size + self.game.grid.tile_size // 2
        agent.y = tile.slot_y * self.game.grid.tile_size + self.game.grid.tile_size // 2
        # Note: slot_x and slot_y are calculated automatically from x and y pixel coordinates
    else:
        # Position agent at nearest walkable tile (should already be there from pathfinding)
        # Clear the path since we've reached the target
        agent.path = []
        agent.path_index = 0
    
    # Process the action based on tile type and agent state
    if hasattr(tile, '_type'):
        if tile._type == 3:  # Dispenser
            handle_dispenser_action(self, agent, tile, action_data)
        elif tile._type == 2:  # Counter
            handle_counter_action(self, agent, tile, action_data)
        elif tile._type == 4:  # Cutting board
            handle_cutting_board_action(self, agent, tile, action_data)
        elif tile._type == 5:  # Delivery
            handle_delivery_action(self, agent, tile, action_data)
    
    # Clear the busy state now that the action is complete
    agent.busy_until = 0

def handle_dispenser_action(self, agent, tile, action_data):
    """Handle picking up items from dispensers."""
    if hasattr(tile, 'item'):
        # Always pick up from dispenser, replacing any item the agent currently has
        agent.item = tile.item
        
    # Position agent next to dispenser (dispensers are not walkable)
    # Agent should already be positioned by pathfinding, but ensure they're stopped
    agent.path = []
    agent.path_index = 0

def handle_counter_action(self, agent, tile, action_data):
    """Handle interactions with counters (placing/picking up items, making salads)."""
    # Get the original action name to determine intent
    action_idx = action_data['action_idx']
    game_mode = getattr(self, 'game_mode', 'classic')
    action_name = get_rl_action_space(game_mode)[action_idx]
    
    # Check if this is a pick_up action
    if action_name.startswith('pick_up_') and 'from_counter' in action_name:
        # Extract the item type from the action name
        # e.g., "pick_up_tomato_cut_from_counter_closest" -> "tomato_cut"
        parts = action_name.split('_')
        if len(parts) >= 6:
            item_to_pick = '_'.join(parts[2:-3])  # Remove "pick", "up", "from", "counter", "closest/midpoint"
            
            # Only proceed if the counter actually has this specific item
            if hasattr(tile, 'item') and tile.item == item_to_pick:
                # Special case: Agent has plate and counter has cut ingredient -> make salad
                if (agent.item == 'plate' and item_to_pick.endswith('_cut')):
                    tile.item = item_to_pick.split('_')[0] + '_salad'
                    if hasattr(tile, 'salad_item'):
                        tile.salad_item = tile.item
                    agent.item = None  # Agent does not pick up the salad
                    
                # Special case: Agent has cut ingredient and counter has plate -> make salad  
                elif (agent.item and agent.item.endswith('_cut') and item_to_pick == 'plate'):
                    salad_item = agent.item.split('_')[0] + '_salad'
                    tile.item = salad_item
                    if hasattr(tile, 'salad_item'):
                        tile.salad_item = salad_item
                    agent.item = None  # Agent does not pick up the salad
                    
                # Normal item exchange
                else:
                    temp_item = tile.item
                    tile.item = agent.item
                    agent.item = temp_item
            # If counter doesn't have the requested item, do nothing (action should have been blocked earlier)
    
    elif action_name.startswith('put_down_item_on_free_counter'):
        # Only proceed if counter is actually free (empty)
        if not hasattr(tile, 'item') or tile.item is None:
            # Place item on the free counter
            tile.item = agent.item
            agent.item = None
        # If counter is not free, do nothing (action should have been blocked earlier)
            
    # Position agent next to counter (counters are not walkable)
    agent.path = []
    agent.path_index = 0

def handle_cutting_board_action(self, agent, tile, action_data):
    """Handle cutting actions on cutting board."""
    if agent.item in ['tomato', 'pumpkin', 'cabbage']:
        # Instantly complete cutting action (as per original intent behavior)
        original_item = agent.item
        cut_item = f"{original_item}_cut"
        agent.item = cut_item
        
        # Update tile metadata
        if hasattr(tile, 'cut_item'):
            tile.cut_item = cut_item
        if hasattr(tile, 'item'):
            tile.item = None  # Clear the cutting board
    
    # Position agent next to cutting board (cutting boards are not walkable)
    # Agent should already be positioned by pathfinding, but ensure they're stopped
    agent.path = []
    agent.path_index = 0

def handle_delivery_action(self, agent, tile, action_data):
    """Handle delivery actions."""
    valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']
    
    if agent.item in valid_items:
        # Set delivery metadata
        if hasattr(tile, 'delivered_by'):
            tile.delivered_by = agent.id
        if hasattr(tile, 'delivered_item'):
            tile.delivered_item = agent.item
            
        # Clear agent item and increment score
        agent.item = None
        agent.score += 1
        
        # Update game score if it exists
        if ('score' in self.game.gameObjects and 
            self.game.gameObjects['score'] is not None):
            self.game.gameObjects['score'].score += 1
    
    # Position agent next to delivery (delivery tiles are not walkable)
    agent.path = []
    agent.path_index = 0

def setup_agent_path(self, agent, tile_index):
    """Set up the agent's movement path to the target tile."""
    
    # Get target tile coordinates
    grid_w = self.game.grid.width
    x = tile_index % grid_w
    y = tile_index // grid_w
    target_tile = self.game.grid.tiles[x][y]
    
    # Set up pathfinding
    start_node = Node(agent.slot_x, agent.slot_y)
    
    # If target tile is walkable, path directly to it
    if hasattr(target_tile, 'is_walkable') and target_tile.is_walkable:
        goal_node = Node(x, y)
    else:
        # For non-walkable tiles, find nearest walkable neighbor
        from engine.extensions.topDownGridWorld.a_star import get_neighbors
        target_node = Node(x, y)
        neighbors = get_neighbors(self.game.grid, target_node, include_diagonal=False)
        
        if not neighbors:
            # No valid path
            agent.path = []
            agent.path_index = 0
            return
            
        # Find closest walkable neighbor
        best_distance = float('inf')
        goal_node = None
        
        for neighbor in neighbors:
            if (0 <= neighbor.x < self.game.grid.width and 
                0 <= neighbor.y < self.game.grid.height):
                neighbor_tile = self.game.grid.tiles[neighbor.x][neighbor.y]
                if hasattr(neighbor_tile, 'is_walkable') and neighbor_tile.is_walkable:
                    # Calculate distance from agent to this neighbor
                    dist = ((agent.slot_x - neighbor.x) ** 2 + (agent.slot_y - neighbor.y) ** 2) ** 0.5
                    if dist < best_distance:
                        best_distance = dist
                        goal_node = neighbor
        
        if goal_node is None:
            # No walkable path found
            agent.path = []
            agent.path_index = 0
            return
    
    # Calculate path
    path = find_path(self.game.grid, start_node, goal_node)
    if path:
        agent.path = path[1:]  # Skip current tile
        agent.path_index = 0
    else:
        agent.path = []
        agent.path_index = 0