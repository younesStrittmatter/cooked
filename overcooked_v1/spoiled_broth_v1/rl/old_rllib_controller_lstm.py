import os
import torch
import numpy as np
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from spoiled_broth.game import game_to_obs_matrix
from engine.extensions.topDownGridWorld.ai_controller._base_controller import Controller

class RLlibControllerLSTM(Controller):
    """
    Controller to load multiple RLlib policies (with LSTM) directly from MultiRLModule checkpoint,
    and use them to select actions for visualization.
    """

    def __init__(self, agent_id, checkpoint_path, policy_id):
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.policy_id = policy_id

        rl_module_path = os.path.join(
            checkpoint_path,
            "learner_group",
            "learner",
            "rl_module"
        )

        print(f"Loading MultiRLModule from: {rl_module_path}")
        self.multi_rl_module = MultiRLModule.from_checkpoint(rl_module_path)

        if self.policy_id not in self.multi_rl_module.keys():
            raise ValueError(f"Policy '{self.policy_id}' not found in checkpoint.")

        self.policy_module = self.multi_rl_module[self.policy_id]

        # Initialize LSTM state if needed
        self.lstm_state = None
        if hasattr(self.policy_module, "get_initial_state"):
            self.lstm_state = self.policy_module.get_initial_state()
            # Show shape/type info
            if isinstance(self.lstm_state, (list, tuple)):
                desc = []
                for s in self.lstm_state:
                    if hasattr(s, 'shape'):
                        desc.append(s.shape)
                    else:
                        desc.append(f"type={type(s).__name__}")
                print(f"Initialized LSTM state for {self.policy_id}: {desc}")
            elif hasattr(self.lstm_state, 'shape'):
                print(f"Initialized LSTM state for {self.policy_id}: single state shape={self.lstm_state.shape}")
            else:
                print(f"Initialized LSTM state for {self.policy_id}: type={type(self.lstm_state).__name__}")

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, (np.ndarray, float, int)):
            return torch.tensor(x)
        elif isinstance(x, dict):
            return {k: self._to_tensor(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._to_tensor(v) for v in x)
        else:
            raise TypeError(f"Unsupported LSTM state type: {type(x)}")

    def choose_action(self, observation=None):
        obs_matrix, agent_inventory = game_to_obs_matrix(self.agent.game, self.agent_id)
        obs_vector = np.concatenate([obs_matrix.flatten(), agent_inventory.flatten()]).astype(np.float32)
        obs_tensor = torch.tensor(obs_vector).unsqueeze(0)  # Shape [1, obs_size]
        print("obs_tensor.shape:", obs_tensor.shape)

        # Helper to add batch dim to LSTM states (if missing)
        def _to_tensor_with_batch(x):
            if isinstance(x, torch.Tensor):
                return x.unsqueeze(0) if x.dim() == 1 else x
            elif isinstance(x, (np.ndarray, float, int)):
                t = torch.tensor(x)
                return t.unsqueeze(0) if t.dim() == 1 else t
            elif isinstance(x, dict):
                return {k: _to_tensor_with_batch(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(_to_tensor_with_batch(v) for v in x)
            else:
                raise TypeError(f"Unsupported LSTM state type: {type(x)}")

        # Initialize or convert LSTM state to dict with actor/critic keys, batched
        if hasattr(self.policy_module, "get_initial_state"):
            if self.lstm_state is None:
                full_state = self.policy_module.get_initial_state()
                half = len(full_state) // 2
                actor_states = full_state[:half]
                critic_states = full_state[half:]
                self.lstm_state = {
                    "actor": _to_tensor_with_batch(actor_states),
                    "critic": _to_tensor_with_batch(critic_states),
                }
            state_in = self.lstm_state
        else:
            state_in = None

        # VERY IMPORTANT: RLlib PPO LSTM expects input_dict["obs"] to be a dict with keys 'actor' and 'critic'
        input_dict = {
            "obs": {
                "actor": obs_tensor,
                "critic": obs_tensor,  # or obs for critic if different
            }
        }
        if state_in is not None:
            input_dict["state_in"] = state_in

        if "state_in" in input_dict:
            for k, v in input_dict["state_in"].items():
                if isinstance(v, (list, tuple)):
                    for i, vv in enumerate(v):
                        print(f" state_in[{k}][{i}]: {vv.shape}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        print(f" state_in[{k}][{kk}]: {vv.shape}")
                else:
                    print(f" state_in[{k}]: {v.shape}")

        # Forward pass
        action_output = self.policy_module.forward_inference(input_dict, seq_lens=torch.tensor([1]))

        logits = action_output["action_dist_inputs"]
        if isinstance(logits, dict):
            logits = logits[self.agent_id]

        # Update LSTM state for next step
        if "state_out" in action_output:
            state_out = action_output["state_out"]
            if isinstance(state_out, dict):
                self.lstm_state = {
                    "actor": state_out.get("actor", self.lstm_state["actor"]),
                    "critic": state_out.get("critic", self.lstm_state["critic"]),
                }
            else:
                self.lstm_state = state_out
        elif any(k.startswith("state_out_") for k in action_output.keys()):
            keys = sorted([k for k in action_output if k.startswith("state_out_")])
            half = len(keys) // 2
            actor_keys = keys[:half]
            critic_keys = keys[half:]
            actor_states = [action_output[k][self.agent_id] if isinstance(action_output[k], dict) else action_output[k] for k in actor_keys]
            critic_states = [action_output[k][self.agent_id] if isinstance(action_output[k], dict) else action_output[k] for k in critic_keys]
            self.lstm_state = {
                "actor": actor_states,
                "critic": critic_states,
            }

        action_dist_cls = self.policy_module.get_inference_action_dist_cls()
        action_dist = action_dist_cls.from_logits(logits)
        action = action_dist.sample().item()

        grid = self.agent.grid
        clickable_indices = getattr(self.agent.game, 'clickable_indices', None)
        if clickable_indices is not None and 0 <= action < len(clickable_indices):
            tile_index = clickable_indices[action]
            x = tile_index % grid.width
            y = tile_index // grid.width
        else:
            x = action % grid.width
            y = action // grid.width
        tile = grid.tiles[x][y]

        if tile and hasattr(tile, "click"):
            return {"type": "click", "target": tile.id}

        return None
