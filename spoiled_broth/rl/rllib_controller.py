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
        # First check if we need to mark the previous action as complete
        if hasattr(self.agent, 'current_action') and not getattr(self.agent, 'action_complete', True):
            # Check if the action is actually complete (e.g. agent has arrived at destination)
            current_action = getattr(self.agent, 'current_action', None)
            if current_action and hasattr(self.agent.game, 'action_tracker'):
                if current_action.get('type') == 'click':
                    # Mark action as complete if agent has finished moving
                    if not self.agent.is_moving:
                        self.agent.game.action_tracker.end_action(self.agent_id, time.time())
                        self.agent.action_complete = True
                        self.agent.current_action = None
                else:
                    # For non-click actions, mark as complete immediately
                    self.agent.game.action_tracker.end_action(self.agent_id, time.time())
                    self.agent.action_complete = True
                    self.agent.current_action = None
            
        # Don't choose a new action if current one is still in progress
        if not self.agent.action_complete:
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
        if clickable_indices is not None:
            # Convert action to -1 for "do nothing" when it's the last action
            action_number = -1 if action == len(clickable_indices) else action
            
            # Handle "do nothing" action
            if action_number == -1:
                if hasattr(self.agent.game, 'action_tracker'):
                    self.agent.game.action_tracker.start_action(
                        self.agent_id,
                        "do_nothing",
                        None,
                        time.time(),
                        action_number=-1
                    )
                self.agent.action_complete = True
                return None
    
            if 0 <= action < len(clickable_indices):
                tile_index = clickable_indices[action]
                grid = self.agent.grid
                x = tile_index % grid.width
                y = tile_index // grid.width
                tile = grid.tiles[x][y]
                if tile and hasattr(tile, "click"):
                    # Record action start
                    if hasattr(self.agent.game, 'action_tracker'):
                        self.agent.game.action_tracker.start_action(
                            self.agent_id,
                            "click",
                            tile,  # Pass the entire tile object instead of just the ID
                            time.time(),
                            action_number=action
                        )
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
                    if hasattr(self.agent.game, 'action_tracker'):
                        self.agent.game.action_tracker.end_action(self.agent_id, time.time())
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
                if hasattr(self.agent.game, 'action_tracker'):
                    self.agent.game.action_tracker.start_action(
                        self.agent_id,
                        "click",
                        tile,  # Pass the full tile object
                        time.time(),
                        action_number=action
                    )
                self.agent.action_complete = False
                self.agent.current_action = {
                    "type": "click", 
                    "target": tile.id,
                    "start_time": time.time()
                }
                return self.agent.current_action
        
        # If we reach here, no valid action was taken
        if hasattr(self.agent.game, 'action_tracker'):
            self.agent.game.action_tracker.end_action(self.agent_id, time.time())
        return None
