#!/usr/bin/env python3
"""
Experimental simulation runner for reinforcement learning experiments.

This script runs simulations of trained RL agents with comprehensive logging,
video recording, and data collection capabilities. After the simulation completes,
it automatically performs meaningful actions analysis to detect and categorize
agent behaviors based on item state changes.

The script automatically saves all output (including meaningful actions analysis)
to a log file in the simulation directory.

Usage:
python experimental_simulation.py <map_nr> <num_agents> <intent_version> <cooperative> <game_version> <training_id> <checkpoint_number> [options]

For background execution:
nohup python experimental_simulation.py <map_nr> <num_agents> <intent_version> <cooperative> <game_version> <training_id> <checkpoint_number> [options] > experimental_simulation.log 2>&1 &

Example:
nohup python experimental_simulation.py simple_kitchen_circular 2 v3.1 1 competition training_001 50 --enable_video true --cluster cuenca --duration 600 --tick_rate 24 --video-fps 24 > experimental_simulation.log 2>&1 &
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.simulations import (
    setup_simulation_argument_parser, 
    main_simulation_pipeline,
    analyze_meaningful_actions,
    generate_agent_position_files,
    generate_agent_action_files
)


class TeeLogger:
    """Class to duplicate output to both console and file"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            self.log_file.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()
            
    def flush(self):
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.flush()


def main():
    """Main execution function."""
    parser = setup_simulation_argument_parser()
    args = parser.parse_args()
    
    # Convert enable_video string to boolean
    enable_video = args.enable_video.lower() == 'true'
    
    # Print initial startup information
    startup_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting experimental simulation at: {startup_time}")
    print(f"Map: {args.map_nr}")
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
            map_nr=args.map_nr,
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
        
        # Setup logging to save in simulation directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = output_paths['simulation_dir'] / f"experimental_simulation_{timestamp}.log"
        
        # Continue with the rest of the execution while logging to file
        with TeeLogger(log_file_path):
            print("\nSimulation completed successfully!")
            print("=" * 50)
            print("Output files:")
            print(f"  Simulation directory: {output_paths['simulation_dir']}")
            print(f"  Configuration file: {output_paths['config_file']}")
            print(f"  State CSV: {output_paths['state_csv']}")
            print(f"  Action CSV: {output_paths['action_csv']}")
            if output_paths['video_file']:
                print(f"  Video file: {output_paths['video_file']}")
            print(f"  Log file: {log_file_path}")
            
            # Run meaningful actions analysis
            print("\n" + "=" * 50)
            print("ANALYZING MEANINGFUL ACTIONS")
            print("=" * 50)
            
            try:
                # Read the generated CSV files
                actions_df = pd.read_csv(output_paths['action_csv'])
                simulation_df = pd.read_csv(output_paths['state_csv'])
                
                # Analyze meaningful actions using the new modular function
                meaningful_df = analyze_meaningful_actions(
                    actions_df, 
                    simulation_df, 
                    args.map_nr, 
                    output_paths['simulation_dir']
                )
                
                if not meaningful_df.empty:
                    # Print summary statistics
                    print(f"\nMEANINGFUL ACTIONS SUMMARY:")
                    print(f"  Total meaningful actions: {len(meaningful_df)}")
                    print(f"  Actions by agent:")
                    for agent_id in meaningful_df['agent_id'].unique():
                        agent_actions = meaningful_df[meaningful_df['agent_id'] == agent_id]
                        print(f"    {agent_id}: {len(agent_actions)} actions")
                    
                    print(f"  Actions by type:")
                    action_counts = meaningful_df['item_change_type'].value_counts()
                    for action_type, count in action_counts.items():
                        print(f"    {action_type}: {count}")
                    
                    print(f"  Top action categories:")
                    category_counts = meaningful_df['action_category_name'].value_counts().head(10)
                    for category, count in category_counts.items():
                        print(f"    {category}: {count}")
                else:
                    print("\nNo meaningful actions detected in this simulation.")
                    
            except Exception as e:
                print(f"\nError during meaningful actions analysis: {e}")
                print("Continuing without meaningful actions analysis...")
            
            # Generate agent position files
            print("\n" + "=" * 50)
            print("GENERATING AGENT POSITION FILES")
            print("=" * 50)
            
            try:
                # Generate position files for each agent
                position_files = generate_agent_position_files(
                    simulation_df, 
                    output_paths['simulation_dir']
                )
                
                print(f"\nAgent position files generated:")
                for agent_id, filepath in position_files.items():
                    print(f"  {agent_id}: {filepath}")
                    
            except Exception as e:
                print(f"\nError during position file generation: {e}")
                print("Continuing without position file generation...")
            
            # Generate agent action files
            print("\n" + "=" * 50)
            print("GENERATING AGENT ACTION FILES")
            print("=" * 50)
            
            try:
                # Generate action files for each agent
                action_files = generate_agent_action_files(
                    meaningful_df,
                    simulation_df, 
                    output_paths['simulation_dir'],
                    map_name=args.map_nr,
                    simulation_id=output_paths['simulation_dir'].name,
                    engine_tick_rate=args.tick_rate
                )
                
                print(f"\nAgent action files generated:")
                for agent_id, filepath in action_files.items():
                    print(f"  {agent_id}: {filepath}")
                    
            except Exception as e:
                print(f"\nError during action file generation: {e}")
                print("Continuing without action file generation...")
            
            print("=" * 50)
            print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Full log saved to: {log_file_path}")
        
    except Exception as e:
        # Try to save error log to simulation directory if possible
        error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_msg = f"Error during simulation at {error_time}: {e}"
        print(error_msg)
        
        # If we have output_paths, try to save error log there
        try:
            if 'output_paths' in locals() and output_paths:
                error_log_path = output_paths['simulation_dir'] / f"experimental_simulation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                with open(error_log_path, 'w') as f:
                    f.write(f"Experimental Simulation Error Log\n")
                    f.write(f"Time: {error_time}\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"Arguments: {vars(args)}\n")
                print(f"Error log saved to: {error_log_path}")
        except Exception as log_error:
            print(f"Could not save error log: {log_error}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()