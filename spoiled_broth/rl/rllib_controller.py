import os
import time
import torch
import numpy as np
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from spoiled_broth.game import game_to_obs_matrix, game_to_obs_matrix_competition
from engine.extensions.topDownGridWorld.ai_controller._base_controller import Controller

class RLlibController(Controller):
    def __init__(self, agent_id, checkpoint_path, policy_id, competition=False):
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.policy_id = policy_id
        self.competition = competition
        # RLModule inside the checkpoint
        rl_module_path = os.path.join(
            checkpoint_path,
            "learner_group",
            "learner",
            "rl_module"
        )

        # Load MultiRLModule from checkpoint
        self.multi_rl_module = MultiRLModule.from_checkpoint(rl_module_path)
        if self.policy_id not in self.multi_rl_module.keys():
            raise ValueError(f'Policy {self.policy_id} not found in the loaded module.')

        # Obtain the specific policy module
        self.policy_module = self.multi_rl_module[self.policy_id]

    def choose_action(self, observation):      
        # If a previous action is still in progress, don't choose a new one.
        # NOTE: Do NOT mutate agent state here; the Engine/Agent update loop is
        # responsible for marking actions complete. Controllers called from an
        # external watcher must be read-only to avoid racing the engine.
        if hasattr(self.agent, 'current_action') and not getattr(self.agent, 'action_complete', True):
            return None
            
        # Don't choose a new action if current one is still in progress
        if not getattr(self.agent, 'action_complete', True):
            return None
        
        # Don't choose a new action if agent is busy (e.g., cutting)
        if getattr(self.agent, 'is_busy', False):
            return None
        
        # Select correct obs function
        self.agent_food_type = {
            "ai_rl_1": "tomato",
            "ai_rl_2": "pumpkin",
        }
        if self.competition:
            obs_matrix, agent_inventory = game_to_obs_matrix_competition(
                self.agent.game, self.agent_id, self.agent_food_type
            )
        else:
            obs_matrix, agent_inventory = game_to_obs_matrix(
                self.agent.game, self.agent_id
            )
        obs_vector = np.concatenate([obs_matrix.flatten(), agent_inventory.flatten()]).astype(np.float32)
        input_dict = {"obs": torch.tensor(obs_vector, dtype=torch.float32)}

        # Obtain the action logits from the policy module
        action_output = self.policy_module.forward_inference(input_dict)
        action_logits = action_output["action_dist_inputs"]

        # Obtain the action distribution class from the policy module
        action_dist_class = self.policy_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(action_logits)
        action = action_dist.sample().item()
        
        # Get clickable indices directly from the game
        clickable_indices = getattr(self.agent.game, 'clickable_indices', None)
        
        # Log the raw action integer to CSV if raw_action_logger exists
        if hasattr(self.agent.game, 'raw_action_logger') and clickable_indices is not None:
            self.agent.game.raw_action_logger.log_action(
                self.agent_id, action, self.agent.game, clickable_indices
            )
        if clickable_indices is not None:
            # Convert action to -1 for "do nothing" when it's the last action
            action_number = -1 if action == len(clickable_indices) else action
            
            # Handle "do nothing" action
            if action_number == -1:
                # Set up action state like other actions
                self.agent.action_complete = False
                self.agent.current_action = {
                    "type": "do_nothing", 
                    "start_time": time.time()
                }
                # Ensure agent is not marked as busy for do_nothing actions
                if hasattr(self.agent, 'is_busy'):
                    self.agent.is_busy = False
                return self.agent.current_action
    
            if 0 <= action < len(clickable_indices):
                tile_index = clickable_indices[action]
                grid = self.agent.grid
                x = tile_index % grid.width
                y = tile_index // grid.width
                tile = grid.tiles[x][y]
                if tile and hasattr(tile, "click"):
                    self.agent.action_complete = False
                    self.agent.current_action = {
                        "type": "click", 
                        "target": tile.id,
                        "start_time": time.time()
                    }
                    return self.agent.current_action
                else:
                    print(f"Warning: Tile at index {tile_index} is invalid or has no click method.")
                    # Mark action as complete if tile is invalid
                    self.agent.action_complete = True
                    self.agent.current_action = None
                    return None

        else:
            # Fallback: treat action as a flat index over the grid (legacy)
            print("Warning: clickable_indices is None, using fallback grid indexing.")
            grid = self.agent.grid
            x = action % grid.width
            y = action // grid.width
            tile = grid.tiles[x][y]
            if tile and hasattr(tile, "click"):
                # Record action start with the full tile object
                self.agent.action_complete = False
                self.agent.current_action = {
                    "type": "click", 
                    "target": tile.id,
                    "start_time": time.time()
                }
                return self.agent.current_action
        
        # If we reach here, no valid action was taken
        return None

    # Compatibility wrapper used by external watchers that try multiple method names/signatures
    def request_action(self, *args, **kwargs):
        """Flexible wrapper so the decision watcher can call this controller with
        different signatures (game, agent_id) or (agent_id, game). We only need
        the game to build observations and the agent_id to identify which agent.
        """
        # Try to extract (game, agent_id) from args
        game = None
        agent_id = None
        # Args may be (game, agent_id) or (agent_id, game) or (game,) or (agent_id,)
        if len(args) >= 2:
            # Heuristic: if first arg has attribute 'grid' it's a game
            if hasattr(args[0], 'grid'):
                game = args[0]
                agent_id = args[1]
            elif hasattr(args[1], 'grid'):
                game = args[1]
                agent_id = args[0]
        elif len(args) == 1:
            # Single argument could be the game or agent_id; prefer game
            if hasattr(args[0], 'grid'):
                game = args[0]
            else:
                agent_id = args[0]

        # Fallback to attributes on the controller
        if game is None:
            game = getattr(self, 'agent', None)
            if game is not None and hasattr(game, 'game'):
                game = game.game
        if agent_id is None:
            agent_id = getattr(self, 'agent_id', None)

        # If we have a game and agent_id, build the observation and delegate
        if game is not None and agent_id is not None:
            # Temporarily set self.agent to the actual agent object for compatibility
            prev_agent = getattr(self, 'agent', None)
            try:
                self.agent = game.gameObjects.get(agent_id)
                return self.choose_action(None)
            finally:
                # restore previous agent
                self.agent = prev_agent

        # Last resort: call choose_action with None
        return self.choose_action(None)
