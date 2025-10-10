"""
Path management utilities for simulation runs.

Author: Samuel Lozano
"""

import os
from pathlib import Path
from typing import Dict, Tuple

from .simulation_config import SimulationConfig


class PathManager:
    """Manages paths and directories for simulation runs."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def setup_paths(self, map_nr: str, num_agents: int, intent_version: str,
                   cooperative: bool, game_version: str, training_id: str,
                   checkpoint_number: str) -> Dict[str, Path]:
        """
        Set up all necessary paths for a simulation run.
        
        Args:
            map_nr: Name of the map
            num_agents: Number of agents
            intent_version: Intent version identifier
            cooperative: Whether this is a cooperative simulation
            game_version: Game version identifier
            training_id: Training identifier
            checkpoint_number: Checkpoint number to load (integer or "final")
            
        Returns:
            Dictionary containing all relevant paths
        """
        # Updated path structure: /data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{map_nr}/simulations/Training_{training_id}/checkpoint_{checkpoint_number}/
        base_path = Path(f"{self.config.local_path}/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{map_nr}")
        
        # New structure: all simulations go under /simulations/Training_{training_id}/checkpoint_{checkpoint_number}/
        simulations_base = base_path / "simulations"
        training_simulations_path = simulations_base / f"Training_{training_id}"
        
        # Validate checkpoint_number
        if checkpoint_number.lower() != "final":
            try:
                int(checkpoint_number)  # Validate it's a number
            except ValueError:
                raise ValueError(f"Invalid checkpoint number: '{checkpoint_number}'. Must be an integer or 'final'")
        
        # Construct paths based on checkpoint type
        checkpoint_simulations_dir = training_simulations_path / f"checkpoint_{checkpoint_number}"
        
        # Training path for model checkpoints (original structure for backward compatibility)
        training_type = "cooperative" if cooperative else "competitive"
        training_path = base_path / f"{training_type}/Training_{training_id}"
        
        if checkpoint_number.lower() == "final":
            checkpoint_dir = training_path / "checkpoint_final"
        else:
            checkpoint_dir = training_path / f"checkpoint_{checkpoint_number}"
        
        # The saving path is now the checkpoint simulations directory where individual simulation folders will be created
        saving_path = checkpoint_simulations_dir
        
        # Find project root by looking for spoiled_broth directory
        project_root = Path(__file__).parent.parent.parent  # path_manager.py is in spoiled_broth/simulations/
        
        paths = {
            'base_path': base_path,
            'training_path': training_path,
            'checkpoint_dir': checkpoint_dir,
            'saving_path': saving_path,
            'simulations_base': simulations_base,
            'training_simulations_path': training_simulations_path,
            'checkpoint_simulations_dir': checkpoint_simulations_dir,
            'path_root': project_root / "spoiled_broth",
            'map_txt_path': project_root / "spoiled_broth" / "maps" / f"{map_nr}.txt",
            'config_path': training_path / "config.txt",
            'checkpoint_number': checkpoint_number  # Keep original checkpoint_number
        }
        
        # Create directories
        os.makedirs(paths['saving_path'], exist_ok=True)
        
        return paths
    
    def get_grid_size_from_map(self, map_txt_path: Path) -> Tuple[int, int]:
        """
        Determine grid size from map file.
        
        Args:
            map_txt_path: Path to map text file
            
        Returns:
            Tuple of (width, height) for the grid
        """
        try:
            with open(map_txt_path) as f:
                lines = [l.rstrip("\n") for l in f.readlines()]
            if lines:
                return (len(lines[0]), len(lines))
        except Exception as e:
            print(f"Warning: Could not read map file {map_txt_path}: {e}")
        
        return self.config.default_grid_size