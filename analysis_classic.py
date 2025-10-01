#!/usr/bin/env python3
"""
Analysis script for classic reinforcement learning experiments.

This script analyzes training results from single-agent experiments,
generating comprehensive visualizations and statistics.

Usage:
    python analysis_classic.py <intent_version> <map_name> [options]

Example:
    python analysis_classic.py v3.1 simple_kitchen_circular --cluster cuenca --smoothing-factor 15
"""

import sys
import os

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import (
    setup_argument_parser, main_analysis_pipeline, MetricDefinitions
)


def generate_classic_plots(analysis_results):
    """Generate all plots specific to classic experiments."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    plotter = analysis_results['plotter']
    
    print("Generating classic experiment plots...")
    
    # Get classic-specific metrics
    metrics = MetricDefinitions.get_classic_metrics()
    
    # Basic plots
    plotter.plot_basic_metrics(df, paths['figures_dir'], "total_deliveries", "Score (deliveries)")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_total", "Pure Reward")
    
    # Individual agent plots  
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "modified_reward_ai_rl_1", "Modified Reward Agent 1")
    
    # Agent-specific metrics plots
    plotter.plot_agent_metrics(df, paths['figures_dir'], metrics['rewarded_metrics_1'], 1)
    
    # Smoothed plots
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "total_deliveries", "Score", by_attitude=False)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_total", "Pure Reward", by_attitude=False)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "modified_reward_ai_rl_1", "Modified Reward Agent 1")
    
    # Smoothed agent metrics
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['rewarded_metrics_1'], 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['movement_metrics_1'], 1, smoothed=True)
    
    print(f"Classic analysis completed. Figures saved to {paths['figures_dir']}")


def main():
    """Main execution function."""
    parser = setup_argument_parser('classic')
    args = parser.parse_args()
    
    print(f"Starting classic experiment analysis...")
    print(f"Intent version: {args.intent_version}")
    print(f"Map: {args.map_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Smoothing factor: {args.smoothing_factor}")
    
    try:
        # Run main analysis pipeline
        analysis_results = main_analysis_pipeline(
            experiment_type='classic',
            intent_version=args.intent_version,
            map_name=args.map_name,
            cluster=args.cluster,
            smoothing_factor=args.smoothing_factor
        )
        
        # Generate classic-specific plots
        generate_classic_plots(analysis_results)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()