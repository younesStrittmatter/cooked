import os
import torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from spoiled_broth.game import game_to_vector
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
        obs_vector = game_to_vector(self.agent.game, self.agent_id)
        input_dict = {"obs": torch.tensor(obs_vector, dtype=torch.float32)}

        # Obtain the action logits from the policy module
        action_output = self.policy_module.forward_inference(input_dict)
        action_logits = action_output["action_dist_inputs"]

        # Obtain the action distribution class from the policy module
        action_dist_class = self.policy_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(action_logits)

        # Sample an action from the distribution
        action = action_dist.sample().item()

        grid = self.agent.grid
        x = action % grid.width
        y = action // grid.width
        tile = grid.tiles[x][y]
        print(f'{self.agent_id} chose tile: ({tile.slot_x}, {tile.slot_y})')

        if tile and hasattr(tile, "click"):
            return {"type": "click", "target": tile.id}
        return None
