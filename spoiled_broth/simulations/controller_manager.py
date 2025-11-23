"""
Controller management utilities for RL simulations.

Author: Samuel Lozano
"""

from pathlib import Path
from typing import Dict, Any

from .simulation_config import SimulationConfig
from .agent_state_manager import AgentStateManager
from spoiled_broth.simulations.rllib_controller import RLlibController

class ControllerManager:
    """Manages RL controller initialization and configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.agent_state_manager = AgentStateManager()
    
    def initialize_controllers(self, num_agents: int, checkpoint_dir: Path,
                             game_version: str, tick_rate: int = 24, 
                             custom_checkpoints: Dict[str, Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Initialize RL controllers for all agents.
        
        Args:
            num_agents: Number of agents to create controllers for
            checkpoint_dir: Directory containing model checkpoints
            game_version: Game version for competition detection
            tick_rate: Simulation tick rate (frames per second)
            custom_checkpoints: Optional checkpoint configuration for custom loading
            
        Returns:
            Dictionary mapping agent IDs to controller instances
        """    
        
        is_competition = game_version.upper() == "COMPETITION"
        
        # Convert initialization period from seconds to frames. Add a small buffer to account for AI/engine synchronization
        agent_initialization_frames = int((self.config.agent_initialization_period + 0.01) * tick_rate)  # +0.01s buffer
        
        controllers = {}
        
        for i in range(1, num_agents + 1):
            agent_id = f"ai_rl_{i}"
            
            # Use custom checkpoint if provided for this agent
            if custom_checkpoints and agent_id in custom_checkpoints and custom_checkpoints[agent_id] is not None:
                checkpoint_info = custom_checkpoints[agent_id]
                if "path" in checkpoint_info and "loaded_agent_id" in checkpoint_info:
                    custom_checkpoint_dir = Path(checkpoint_info["path"])
                    policy_id = checkpoint_info["loaded_agent_id"]
                    checkpoint_number = checkpoint_info['checkpoint_number']
                    controller_checkpoint_dir = Path(custom_checkpoint_dir) / f"checkpoint_{checkpoint_number}"
                    print(f"Using custom checkpoint for {agent_id}: {controller_checkpoint_dir} with policy {policy_id}")
            else:
                controller_checkpoint_dir = checkpoint_dir
                policy_id = f"policy_{agent_id}"

            try:
                controller = RLlibController(
                    agent_id, 
                    controller_checkpoint_dir, 
                    policy_id, 
                    competition=is_competition,
                    agent_initialization_frames=agent_initialization_frames
                )
                # Add state manager to controller
                controller.agent_state_manager = self.agent_state_manager
                controllers[agent_id] = controller
                print(f"Successfully initialized controller for {agent_id} (init frames: {agent_initialization_frames})")
            except Exception as e:
                print(f"Warning: couldn't initialize controller {agent_id}: {e}")
        
        return controllers