#!/usr/bin/env python3
"""
Test script for the new simulation system.

This script provides a simple way to test the simulation utilities
without running a full simulation.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.simulations.utils import (
    SimulationConfig,
    PathManager,
    setup_simulation_argument_parser
)


def test_config():
    """Test the SimulationConfig class."""
    print("Testing SimulationConfig...")
    
    config = SimulationConfig()
    print(f"  Default cluster: {config.cluster}")
    print(f"  Default duration: {config.duration_seconds} seconds")
    print(f"  Total frames: {config.total_frames}")
    print(f"  Local path: {config.local_path}")
    
    # Test validation
    try:
        config.cluster = 'invalid'
        config.validate_cluster()
        print("  ERROR: Should have failed validation")
    except ValueError as e:
        print(f"  ✓ Validation working: {e}")
    
    print("  ✓ SimulationConfig test passed\n")


def test_path_manager():
    """Test the PathManager class."""
    print("Testing PathManager...")
    
    config = SimulationConfig()
    path_manager = PathManager(config)
    
    # Test path setup (dry run - don't create directories)
    paths = path_manager.setup_paths(
        map_name="test_map",
        num_agents=2,
        intent_version="v3.1",
        cooperative=True,
        game_version="competition",
        training_id="test_training",
        checkpoint_number=50
    )
    
    print(f"  Base path: {paths['base_path']}")
    print(f"  Training path: {paths['training_path']}")
    print(f"  Checkpoint dir: {paths['checkpoint_dir']}")
    print(f"  Saving path: {paths['saving_path']}")
    
    print("  ✓ PathManager test passed\n")


def test_data_logger():
    """Test the DataLogger class."""
    print("Testing DataLogger...")
    
    import tempfile
    import time
    from pathlib import Path
    from spoiled_broth.simulations.utils import DataLogger
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        
        simulation_config = {
            'map_name': 'test_map',
            'num_agents': 2,
            'intent_version': 'v3.1',
            'cooperative': True,
            'game_version': 'competition',
            'training_id': 'test_training',
            'cluster': 'cuenca',
            'duration': 180,
            'tick_rate': 24,
            'enable_video': True,
            'video_fps': 24
        }
        
        logger = DataLogger(temp_path, 50, timestamp, simulation_config)
        
        print(f"  Simulation dir: {logger.simulation_dir}")
        print(f"  Config file: {logger.config_path}")
        print(f"  State CSV: {logger.state_csv_path}")
        print(f"  Action CSV: {logger.action_csv_path}")
        
        # Check that directory was created
        assert logger.simulation_dir.exists(), "Simulation directory not created"
        assert logger.config_path.exists(), "Config file not created"
        assert logger.state_csv_path.exists(), "State CSV not created"
        
        # Verify config file content
        with open(logger.config_path) as f:
            config_content = f.read()
            assert 'SIMULATION_ID' in config_content, "Config missing simulation ID"
            assert 'test_map' in config_content, "Config missing map name"
            assert 'competition' in config_content, "Config missing game version"
        
        print("  ✓ Config file created successfully")
        print("  ✓ Directory structure created correctly")
    
    print("  ✓ DataLogger test passed\n")


def test_argument_parser():
    """Test the argument parser."""
    print("Testing argument parser...")
    
    parser = setup_simulation_argument_parser()
    
    # Test with minimal arguments
    test_args = [
        "test_map", "2", "v3.1", "1", "competition", "test_training", "50"
    ]
    
    args = parser.parse_args(test_args)
    
    print(f"  Map name: {args.map_name}")
    print(f"  Num agents: {args.num_agents}")
    print(f"  Intent version: {args.intent_version}")
    print(f"  Cooperative: {args.cooperative}")
    print(f"  Game version: {args.game_version}")
    print(f"  Training ID: {args.training_id}")
    print(f"  Checkpoint: {args.checkpoint_number}")
    print(f"  Enable video: {args.enable_video}")
    print(f"  Cluster: {args.cluster}")
    
    print("  ✓ Argument parser test passed\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Simulation System Components")
    print("=" * 50)
    
    try:
        test_config()
        test_path_manager()
        test_data_logger()
        test_argument_parser()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("The simulation system is ready to use.")
        print("=" * 50)
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()