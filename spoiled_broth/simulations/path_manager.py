"""
Path management utilities for simulation runs.

Author: Samuel Lozano
"""

import os
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

from .simulation_config import SimulationConfig


class PathManager:
    """Manages paths and directories for simulation runs."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config

    def _find_last_epoch_in_training_stats(self, training_path: Path) -> Optional[int]:
        """
        Find the last epoch number in the training_stats folder.
        
        Args:
            training_path: Path to the training folder
            training_id: Training identifier
            
        Returns:
            Last epoch number as string, or None if not found
        """
        training_stats_path = training_path / "training_stats.csv"
        
        if not training_stats_path.exists():
            return None
    
        max_epoch = -1

        try:
            with open(training_stats_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                if 'epoch' not in reader.fieldnames:
                    print(f"Warning: 'epoch' column not found in {training_stats_path}")
                    return None

                for row in reader:
                    try:
                        epoch_value = int(row['epoch'])
                        if epoch_value > max_epoch:
                            max_epoch = epoch_value
                    except ValueError:
                        continue
                    
            if max_epoch >= 0:
                return max_epoch

        except Exception as e:
            print(f"Warning: Could not read or process training_stats file {training_stats_path}: {e}")

        return None

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

        # Training path for model checkpoints (original structure for backward compatibility)
        training_type = "cooperative" if cooperative else "competitive"
        training_path = base_path / f"{training_type}/Training_{training_id}"
        checkpoint_dir = training_path / f"checkpoint_{checkpoint_number}"
        
        # Handle "final" checkpoint by finding the last epoch
        checkpoint_number_for_folder = checkpoint_number
        if checkpoint_number.lower() == "final":            
            last_epoch = self._find_last_epoch_in_training_stats(training_path)
            if last_epoch is not None:
                checkpoint_number_for_folder = str(last_epoch + 1)
                print(f"Found last epoch {checkpoint_number_for_folder} for final checkpoint")
            else:
                print(f"Warning: Could not find last epoch in training_stats, using 'final' for folder name")
                checkpoint_number_for_folder = "final"
        
        # Validate checkpoint_number (only if it's not "final")
        if checkpoint_number.lower() != "final":
            try:
                int(checkpoint_number)  # Validate it's a number
            except ValueError:
                raise ValueError(f"Invalid checkpoint number: '{checkpoint_number}'. Must be an integer or 'final'")
        
        # Construct paths based on checkpoint type - use checkpoint_number_for_folder for the folder name
        checkpoint_simulations_dir = training_simulations_path / f"checkpoint_{checkpoint_number_for_folder}"
        
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
            'checkpoint_number': checkpoint_number,  # Keep original checkpoint_number
            'checkpoint_number_for_folder': checkpoint_number_for_folder  # Actual number used for folder name
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