#!/usr/bin/env python3
"""
Experimental simulation runner for reinforcement learning experiments.

This script runs simulations of trained RL agents with comprehensive logging,
video recording, and data collection capabilities.

Usage:
    python experimental_simulation_new.py <map_name> <num_agents> <intent_version> <cooperative> <game_version> <training_id> <checkpoint_number> [options]

Example:
    python experimental_simulation_new.py simple_kitchen_circular 2 v3.1 1 competition training_001 50 --enable-video true --cluster cuenca
"""

import sys
import os

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.simulations.utils import (
    setup_simulation_argument_parser, 
    main_simulation_pipeline
)


def main():
    """Main execution function."""
    parser = setup_simulation_argument_parser()
    args = parser.parse_args()
    
    # Convert enable_video string to boolean
    enable_video = args.enable_video.lower() == 'true'
    
    print(f"Starting experimental simulation...")
    print(f"Map: {args.map_name}")
    print(f"Agents: {args.num_agents}")
    print(f"Intent version: {args.intent_version}")
    print(f"Cooperative: {'Yes' if args.cooperative else 'No'}")
    print(f"Game version: {args.game_version}")
    print(f"Training ID: {args.training_id}")
    print(f"Checkpoint: {args.checkpoint_number}")
    print(f"Video recording: {'Enabled' if enable_video else 'Disabled'}")
    print(f"Cluster: {args.cluster}")
    print(f"Duration: {args.duration} seconds")
    print(f"Tick rate: {args.tick_rate} FPS")
    
    try:
        # Run main simulation pipeline
        output_paths = main_simulation_pipeline(
            map_name=args.map_name,
            num_agents=args.num_agents,
            intent_version=args.intent_version,
            cooperative=bool(args.cooperative),
            game_version=args.game_version,
            training_id=args.training_id,
            checkpoint_number=args.checkpoint_number,
            enable_video=enable_video,
            cluster=args.cluster,
            duration=args.duration,
            tick_rate=args.tick_rate,
            video_fps=args.video_fps
        )
        
        print("\nSimulation completed successfully!")
        print("=" * 50)
        print("Output files:")
        print(f"  Simulation directory: {output_paths['simulation_dir']}")
        print(f"  Configuration file: {output_paths['config_file']}")
        print(f"  State CSV: {output_paths['state_csv']}")
        print(f"  Action CSV: {output_paths['action_csv']}")
        if output_paths['video_file']:
            print(f"  Video file: {output_paths['video_file']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()