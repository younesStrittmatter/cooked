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
                             game_version: str, tick_rate: int = 24) -> Dict[str, Any]:
        """
        Initialize RL controllers for all agents.
        
        Args:
            num_agents: Number of agents to create controllers for
            checkpoint_dir: Directory containing model checkpoints
            game_version: Game version for competition detection
            tick_rate: Simulation tick rate (frames per second)
            
        Returns:
            Dictionary mapping agent IDs to controller instances
        """    
        
        is_competition = game_version.upper() == "COMPETITION"
        
        # Convert initialization period from seconds to frames. Add a small buffer to account for AI/engine synchronization
        agent_initialization_frames = int((self.config.agent_initialization_period) * tick_rate)
        
        controllers = {}
        
        for i in range(1, num_agents + 1):
            agent_id = f"ai_rl_{i}"
            try:
                controller = RLlibController(
                    agent_id, 
                    checkpoint_dir, 
                    f"policy_{agent_id}", 
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