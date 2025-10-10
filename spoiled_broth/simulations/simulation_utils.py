"""
Utility functions for simulation pipeline.

Author: Samuel Lozano
"""

import time
import argparse
from pathlib import Path
from typing import Dict

from .simulation_config import SimulationConfig
from .simulation_runner import SimulationRunner


def setup_simulation_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser for simulations.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Run reinforcement learning simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'map_nr',
        type=str,
        help='Map name identifier'
    )
    
    parser.add_argument(
        'num_agents',
        type=int,
        help='Number of agents in the simulation'
    )
    
    parser.add_argument(
        'intent_version',
        type=str,
        help='Intent version identifier'
    )
    
    parser.add_argument(
        'cooperative',
        type=int,
        choices=[0, 1],
        help='Whether simulation is cooperative (1) or competitive (0)'
    )
    
    parser.add_argument(
        'game_version',
        type=str,
        help='Game version identifier'
    )
    
    parser.add_argument(
        'training_id',
        type=str,
        help='Training identifier'
    )
    
    parser.add_argument(
        'checkpoint_number',
        type=str,
        help='Checkpoint number to load (integer) or "final" for the latest checkpoint'
    )
    
    parser.add_argument(
        '--enable-video',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Enable video recording'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        choices=['brigit', 'cuenca', 'local'],
        default='cuenca',
        help='Cluster type for path configuration'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=180,
        help='Simulation duration in seconds'
    )

    parser.add_argument(
        '--agent_initialization_period',
        type=float,
        default=15.0,
        help='Agent initialization period in seconds (no actions taken during this time)'
    )
    
    parser.add_argument(
        '--tick-rate',
        type=int,
        default=24,
        help='Engine tick rate (frames per second)'
    )
    
    parser.add_argument(
        '--video-fps',
        type=int,
        default=24,
        help='Video recording frame rate'
    )
    
    return parser


def main_simulation_pipeline(map_nr: str, num_agents: int, intent_version: str,
                           cooperative: bool, game_version: str, training_id: str,
                           checkpoint_number: str, enable_video: bool = True,
                           cluster: str = 'cuenca', duration: int = 180,
                           tick_rate: int = 24, video_fps: int = 24,
                           agent_initialization_period: float = 15.0) -> Dict[str, Path]:
    """
    Main simulation pipeline that can be used by different simulation scripts.
    
    Args:
        MAP_NR: Map name identifier
        NUM_AGENTS: Number of agents
        intent_version: Intent version identifier
        cooperative: Whether simulation is cooperative
        game_version: Game version identifier
        training_id: Training identifier
        checkpoint_number: Checkpoint number (integer or "final")
        enable_video: Whether to enable video recording
        cluster: Cluster type
        duration: Simulation duration in seconds
        tick_rate: Engine tick rate
        video_fps: Video frame rate
        
    Returns:
        Dictionary containing output file paths
    """
    # Create configuration
    config = SimulationConfig(
        cluster=cluster,
        engine_tick_rate=tick_rate,
        duration_seconds=duration,
        enable_video=enable_video,
        video_fps=video_fps,
        agent_initialization_period=agent_initialization_period
    )
    
    # Create timestamp
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    
    # Create and run simulation
    runner = SimulationRunner(config)
    
    output_paths = runner.run_simulation(
        map_nr=map_nr,
        num_agents=num_agents,
        intent_version=intent_version,
        cooperative=cooperative,
        game_version=game_version,
        training_id=training_id,
        checkpoint_number=checkpoint_number,
        timestamp=timestamp
    )
    
    return output_paths