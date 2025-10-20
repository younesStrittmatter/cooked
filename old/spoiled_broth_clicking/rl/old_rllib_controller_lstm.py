from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
import torch


class RLlibControllerLSTM:
    def __init__(self, agent_id, checkpoint_path, policy_id, competition=False, agent_initialization_frames=0):
        """
        LSTM-capable controller using RLlib checkpoint.
        Args:
            agent_id (str): The agent's ID (e.g., "ai_rl_1")
            checkpoint_path (Path or str): Path to RLlib checkpoint directory
            policy_id (str): Policy ID from training (e.g., "policy_ai_rl_1")
            competition (bool): Whether this is a competition game
            agent_initialization_frames (int): Initialization period in frames
        """
        self.agent_id = agent_id
        self.checkpoint_path = str(checkpoint_path)
        self.policy_id = policy_id
        self.competition = competition
        self.agent_initialization_frames = agent_initialization_frames

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

    def choose_action(self, observation):
        """Choose action using LSTM policy with initialization period check."""
        # Check if we're still in initialization period using frame count
        if self.agent_initialization_frames > 0:
            if hasattr(self.agent, 'game'):
                game = self.agent.game
                
                # Get frame count from game for timing
                if hasattr(game, 'frame_count'):
                    frame_count = getattr(game, 'frame_count')
                    
                    if frame_count < self.agent_initialization_frames:
                        # During initialization, return None (no action)
                        return None
        
        # If a previous action is still in progress, don't choose a new one.
        if hasattr(self.agent, 'current_action') and not getattr(self.agent, 'action_complete', True):
            return None
            
        # Use the existing compute_action method for the actual decision
        return self.compute_action(observation)

    def compute_action(self, obs):
        """Compute action from current observation and LSTM state."""
        action, self.lstm_state, _ = self.policy.compute_single_action(
            obs,
            state=self.lstm_state,
            seq_lens=[1],
            explore=False
        )
        return action