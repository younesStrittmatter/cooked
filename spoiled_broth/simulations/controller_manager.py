"""
Controller management utilities for RL simulations.

Author: Samuel Lozano
"""

from pathlib import Path
from typing import Dict, Any

from .simulation_config import SimulationConfig


class ControllerManager:
    """Manages RL controller initialization and configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def determine_controller_type(self, config_path: Path) -> str:
        """
        Determine which controller type to use based on config file.
        
        Args:
            config_path: Path to training config file
            
        Returns:
            Controller type ('lstm' or 'standard')
        """
        use_lstm = False
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    for line in f:
                        if line.strip().startswith("USE_LSTM"):
                            use_lstm = line.split(":", 1)[1].strip().lower() == "true"
                            break
            except Exception as e:
                print(f"Warning: Could not read config file {config_path}: {e}")
        
        return 'lstm' if use_lstm else 'standard'
    
    def initialize_controllers(self, num_agents: int, checkpoint_dir: Path,
                             controller_type: str, game_version: str) -> Dict[str, Any]:
        """
        Initialize RL controllers for all agents.
        
        Args:
            num_agents: Number of agents to create controllers for
            checkpoint_dir: Directory containing model checkpoints
            controller_type: Type of controller ('lstm' or 'standard')
            game_version: Game version for competition detection
            
        Returns:
            Dictionary mapping agent IDs to controller instances
        """
        # Import controllers dynamically to avoid import issues
        from spoiled_broth.rl.rllib_controller import RLlibController
        from spoiled_broth.rl.old_rllib_controller_lstm import RLlibControllerLSTM
        
        ControllerCls = RLlibControllerLSTM if controller_type == 'lstm' else RLlibController
        is_competition = game_version.upper() == "COMPETITION"
        
        controllers = {}
        
        for i in range(1, num_agents + 1):
            agent_id = f"ai_rl_{i}"
            try:
                controllers[agent_id] = ControllerCls(
                    agent_id, 
                    checkpoint_dir, 
                    f"policy_{agent_id}", 
                    competition=is_competition
                )
                print(f"Successfully initialized controller for {agent_id}")
            except Exception as e:
                print(f"Warning: couldn't initialize controller {agent_id}: {e}")
        
        return controllers