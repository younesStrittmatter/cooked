import os
import torch
import numpy as np
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from spoiled_broth.game import game_to_obs_matrix
from engine.extensions.topDownGridWorld.ai_controller._base_controller import Controller

class RLlibController(Controller):
    def __init__(self, agent_id, checkpoint_path, policy_id):
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.policy_id = policy_id
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
        # Use the new observation space: flatten (channels, H, W) + (2, 4) inventory
        obs_matrix, agent_inventory = game_to_obs_matrix(self.agent.game, self.agent_id)
        obs_vector = np.concatenate([obs_matrix.flatten(), agent_inventory.flatten()]).astype(np.float32)
        input_dict = {"obs": torch.tensor(obs_vector, dtype=torch.float32)}

        # Obtain the action logits from the policy module
        action_output = self.policy_module.forward_inference(input_dict)
        action_logits = action_output["action_dist_inputs"]

        # Obtain the action distribution class from the policy module
        action_dist_class = self.policy_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(action_logits)
        action = action_dist.sample().item()
        
        # Get clickable indices from environment
        clickable_indices = getattr(self.agent.game, 'clickable_indices', None)
        if clickable_indices is not None:
            # Check if action is the last index: do nothing
            if action == len(clickable_indices):
                # The agent chose the "do nothing" action
                return None
    
            if 0 <= action < len(clickable_indices):
                tile_index = clickable_indices[action]
                grid = self.agent.grid
                x = tile_index % grid.width
                y = tile_index // grid.width
                tile = grid.tiles[x][y]
                if tile and hasattr(tile, "click"):
                    return {"type": "click", "target": tile.id}
        else:
            # Fallback: treat action as a flat index over the grid (legacy)
            grid = self.agent.grid
            x = action % grid.width
            y = action // grid.width
            tile = grid.tiles[x][y]
            if tile and hasattr(tile, "click"):
                return {"type": "click", "target": tile.id}
    
        return None
