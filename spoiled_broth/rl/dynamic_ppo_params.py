"""
Dynamic PPO parameters module for implementing exponential decay in PPO hyperparameters during training.
"""
import math
import copy
from typing import Dict, Any


def calculate_dynamic_ppo_params(
    episode: int,
    initial_ppo_cfg: Dict[str, float],
    dynamic_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate dynamically adjusted PPO parameters using exponential decay.
    
    Args:
        episode: Current episode number
        initial_ppo_cfg: Initial PPO parameters configuration
        dynamic_config: Configuration for dynamic PPO params including:
            - enabled: Whether dynamic PPO params are enabled
            - decay_rate: Rate of exponential decay
            - min_param_multiplier: Minimum multiplier for parameters (0-1)
            - decay_start_episode: Episode to start applying decay
            - affected_params: List of parameter names to apply decay to
    
    Returns:
        Updated PPO configuration with exponentially decayed values
    """
    if not dynamic_config.get("enabled", False):
        return initial_ppo_cfg.copy()
    
    # Start with a copy of initial parameters
    updated_params = initial_ppo_cfg.copy()
    
    # Only apply decay after the start episode
    decay_start = dynamic_config.get("decay_start_episode", 0)
    if episode < decay_start:
        return updated_params
    
    # Calculate episodes since decay started
    episodes_since_start = episode - decay_start
    
    # Get decay parameters
    decay_rate = dynamic_config.get("decay_rate", 0.0001)
    min_multiplier = dynamic_config.get("min_param_multiplier", 0.1)
    affected_params = dynamic_config.get("affected_params", [])
    
    # Calculate exponential decay multiplier
    # Formula: multiplier = exp(-decay_rate * episodes_since_start)
    # Ensure it doesn't go below min_multiplier
    decay_multiplier = max(
        min_multiplier,
        math.exp(-decay_rate * episodes_since_start)
    )
    
    # Apply decay to specified parameter types
    for param_name in affected_params:
        if param_name in updated_params:
            updated_params[param_name] = initial_ppo_cfg[param_name] * decay_multiplier
    
    return updated_params


def get_ppo_decay_info(
    episode: int,
    dynamic_config: Dict[str, Any],
    param_name: str = "clip_eps"
) -> Dict[str, float]:
    """
    Get information about the current decay status for PPO parameters.
    
    Args:
        episode: Current episode number
        dynamic_config: Dynamic PPO parameters configuration
        param_name: Parameter name to get info for (for reference)
    
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
    min_multiplier = dynamic_config.get("min_param_multiplier", 0.1)
    
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


class DynamicPPOParamsWrapper:
    """
    Wrapper for managing dynamically adjusted PPO parameters
    based on the current episode and exponential decay parameters.
    """
    
    def __init__(self, initial_ppo_cfg: Dict[str, float], dynamic_config: Dict[str, Any]):
        """
        Initialize the wrapper.
        
        Args:
            initial_ppo_cfg: Initial PPO parameters configuration
            dynamic_config: Dynamic PPO parameters configuration
        """
        self.initial_ppo_cfg = initial_ppo_cfg.copy()
        self.dynamic_config = dynamic_config
        self.current_episode = 0
        
        # Set initial parameters
        self.current_ppo_cfg = self.initial_ppo_cfg.copy()
    
    def update_params_for_episode(self, episode: int):
        """
        Update the PPO parameters for the given episode.
        
        Args:
            episode: Current episode number
        """
        self.current_episode = episode
        self.current_ppo_cfg = calculate_dynamic_ppo_params(
            episode, 
            self.initial_ppo_cfg, 
            self.dynamic_config
        )
        
        # Log decay info if enabled and decay is active
        if self.dynamic_config.get("enabled", False):
            decay_info = get_ppo_decay_info(episode, self.dynamic_config)
            if decay_info["decay_active"] and episode % 100 == 0:  # Log every 100 episodes
                affected_params = self.dynamic_config.get("affected_params", [])
                print(f"[Episode {episode}] PPO parameters decay multiplier: {decay_info['multiplier']:.4f}")
                for param_name in affected_params:
                    if param_name in self.current_ppo_cfg:
                        initial_val = self.initial_ppo_cfg[param_name]
                        current_val = self.current_ppo_cfg[param_name]
                        print(f"  {param_name}: {initial_val:.4f} -> {current_val:.4f}")
    
    def get_current_params(self) -> Dict[str, float]:
        """Get the current PPO parameters configuration."""
        return self.current_ppo_cfg.copy()
    
    def get_decay_status(self) -> Dict[str, Any]:
        """Get current decay status information."""
        return get_ppo_decay_info(self.current_episode, self.dynamic_config)