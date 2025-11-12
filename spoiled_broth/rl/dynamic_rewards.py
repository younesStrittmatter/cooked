"""
Dynamic rewards module for implementing exponential decay in reward values during training.
"""
import math
import copy
from typing import Dict, List, Any


def calculate_dynamic_rewards(
    episode: int,
    initial_rewards_cfg: Dict[str, float],
    dynamic_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate dynamically adjusted rewards using exponential decay.
    
    Args:
        episode: Current episode number
        initial_rewards_cfg: Initial reward configuration
        dynamic_config: Configuration for dynamic rewards including:
            - enabled: Whether dynamic rewards are enabled
            - decay_rate: Rate of exponential decay
            - min_reward_multiplier: Minimum multiplier for rewards (0-1)
            - decay_start_episode: Episode to start applying decay
            - affected_rewards: List of reward types to apply decay to
    
    Returns:
        Updated rewards configuration with exponentially decayed values
    """
    if not dynamic_config.get("enabled", False):
        return initial_rewards_cfg.copy()
    
    # Start with a copy of initial rewards
    updated_rewards = initial_rewards_cfg.copy()
    
    # Only apply decay after the start episode
    decay_start = dynamic_config.get("decay_start_episode", 0)
    if episode < decay_start:
        return updated_rewards
    
    # Calculate episodes since decay started
    episodes_since_start = episode - decay_start
    
    # Get decay parameters
    decay_rate = dynamic_config.get("decay_rate", 0.0001)
    min_multiplier = dynamic_config.get("min_reward_multiplier", 0.1)
    affected_rewards = dynamic_config.get("affected_rewards", [])
    
    # Calculate exponential decay multiplier
    # Formula: multiplier = exp(-decay_rate * episodes_since_start)
    # Ensure it doesn't go below min_multiplier
    decay_multiplier = max(
        min_multiplier,
        math.exp(-decay_rate * episodes_since_start)
    )
    
    # Apply decay to specified reward types
    for reward_type in affected_rewards:
        if reward_type in updated_rewards:
            updated_rewards[reward_type] = initial_rewards_cfg[reward_type] * decay_multiplier
    
    return updated_rewards


def get_decay_info(
    episode: int,
    dynamic_config: Dict[str, Any],
    reward_type: str = "deliver"
) -> Dict[str, float]:
    """
    Get information about the current decay status.
    
    Args:
        episode: Current episode number
        dynamic_config: Dynamic rewards configuration
        reward_type: Reward type to get info for (for reference)
    
    Returns:
        Dictionary with decay information including current multiplier
    """
    if not dynamic_config.get("enabled", False):
        return {"multiplier": 1.0, "decay_active": False}
    
    decay_start = dynamic_config.get("decay_start_episode", 0)
    if episode < decay_start:
        return {"multiplier": 1.0, "decay_active": False}
    
    episodes_since_start = episode - decay_start
    decay_rate = dynamic_config.get("decay_rate", 0.0001)
    min_multiplier = dynamic_config.get("min_reward_multiplier", 0.1)
    
    decay_multiplier = max(
        min_multiplier,
        math.exp(-decay_rate * episodes_since_start)
    )
    
    return {
        "multiplier": decay_multiplier,
        "decay_active": True,
        "episodes_since_start": episodes_since_start,
        "decay_rate": decay_rate,
        "min_multiplier": min_multiplier
    }


class DynamicRewardsWrapper:
    """
    Wrapper for the GameEnv that dynamically updates rewards configuration
    based on the current episode and exponential decay parameters.
    """
    
    def __init__(self, env, initial_rewards_cfg: Dict[str, float], dynamic_config: Dict[str, Any]):
        """
        Initialize the wrapper.
        
        Args:
            env: The GameEnv instance to wrap
            initial_rewards_cfg: Initial reward configuration
            dynamic_config: Dynamic rewards configuration
        """
        self.env = env
        self.initial_rewards_cfg = initial_rewards_cfg.copy()
        self.dynamic_config = dynamic_config
        self.current_episode = 0
        
        # Set initial rewards
        self.env.rewards_cfg = self.initial_rewards_cfg.copy()
    
    def update_rewards_for_episode(self, episode: int):
        """
        Update the environment's rewards configuration for the given episode.
        
        Args:
            episode: Current episode number
        """
        self.current_episode = episode
        updated_rewards = calculate_dynamic_rewards(
            episode, 
            self.initial_rewards_cfg, 
            self.dynamic_config
        )
        self.env.rewards_cfg = updated_rewards
        
        # Log decay info if enabled and decay is active
        if self.dynamic_config.get("enabled", False):
            decay_info = get_decay_info(episode, self.dynamic_config)
            if decay_info["decay_active"] and episode % 100 == 0:  # Log every 100 episodes
                print(f"[Episode {episode}] Reward decay multiplier: {decay_info['multiplier']:.4f}")
    
    def get_current_rewards(self) -> Dict[str, float]:
        """Get the current rewards configuration."""
        return self.env.rewards_cfg.copy()
    
    def get_decay_status(self) -> Dict[str, Any]:
        """Get current decay status information."""
        return get_decay_info(self.current_episode, self.dynamic_config)