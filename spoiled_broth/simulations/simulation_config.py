"""
Configuration class for simulation parameters.

Author: Samuel Lozano
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    # Cluster settings
    cluster: str = 'cuenca'
    cluster_paths: Dict[str, str] = None
    
    # Simulation timing
    engine_tick_rate: int = 24
    ai_tick_rate: int = 1
    duration_seconds: int = 180
    agent_speed_px_per_sec: int = 30
    agent_initialization_period: float = 0.0  # Agent initialization period in seconds
    
    # Video settings
    enable_video: bool = True
    video_fps: int = 24
    
    # Grid settings
    default_grid_size: Tuple[int, int] = (8, 8)
    tile_size: int = 16
    
    # Agent speeds
    walking_speeds: Dict[str, float] = None
    cutting_speeds: Dict[str, float] = None
    
    # Checkpoint configuration
    custom_checkpoints: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.cluster_paths is None:
            self.cluster_paths = {
                'brigit': '/mnt/lustre/home/samuloza',
                'cuenca': '',
                'local': 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
            }
        
        # Initialize speed dictionaries with defaults if not provided
        if self.walking_speeds is None:
            self.walking_speeds = {}
        if self.cutting_speeds is None:
            self.cutting_speeds = {}
        if self.custom_checkpoints is None:
            self.custom_checkpoints = {}
    
    def validate_cluster(self):
        """Validate cluster configuration."""
        if self.cluster not in self.cluster_paths:
            raise ValueError(f"Invalid cluster '{self.cluster}'. Choose from {list(self.cluster_paths.keys())}")
    
    @property
    def local_path(self) -> str:
        """Get the local path for the configured cluster."""
        self.validate_cluster()
        return self.cluster_paths[self.cluster]
    
    @property
    def total_frames(self) -> int:
        """Calculate total frames for the simulation (including initialization period)."""
        total_simulation_time = self.duration_seconds + self.agent_initialization_period
        return int(total_simulation_time * self.engine_tick_rate)
    
    @property
    def total_simulation_time(self) -> float:
        """Get total simulation time including initialization period."""
        return self.duration_seconds + self.agent_initialization_period