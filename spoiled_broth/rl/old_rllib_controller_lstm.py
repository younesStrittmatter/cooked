from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
import torch


class RLlibControllerLSTM:
    def __init__(self, agent_id, checkpoint_path, policy_id):
        """
        LSTM-capable controller using RLlib checkpoint.
        Args:
            agent_id (str): The agent's ID (e.g., "ai_rl_1")
            checkpoint_path (Path or str): Path to RLlib checkpoint directory
            policy_id (str): Policy ID from training (e.g., "policy_ai_rl_1")
        """
        self.agent_id = agent_id
        self.checkpoint_path = str(checkpoint_path)
        self.policy_id = policy_id

        # --- Load model from checkpoint ---
        self.algo: Algorithm = Algorithm.from_checkpoint(self.checkpoint_path)

        # --- Force policy to run on CPU ---
        self.policy: Policy = self.algo.get_policy(self.policy_id)
        self.policy.model.to("cpu")

        if not self.policy.is_recurrent():
            raise ValueError("This controller requires a recurrent (LSTM) policy.")

        # --- Initialize LSTM state [hidden, cell] ---
        self.lstm_state = self.policy.get_initial_state()

    def reset(self):
        """Reset the internal LSTM state (for new episode)."""
        self.lstm_state = self.policy.get_initial_state()

    def compute_action(self, obs):
        """Compute action from current observation and LSTM state."""
        action, self.lstm_state, _ = self.policy.compute_single_action(
            obs,
            state=self.lstm_state,
            seq_lens=[1],
            explore=False
        )
        return action