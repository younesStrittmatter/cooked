#!/usr/bin/env python3
"""
Analysis script for classic reinforcement learning experiments.

This script analyzes training results from single-agent experiments,
generating comprehensive visualizations and statistics.

Usage:
nohup python analysis_classic.py <intent_version> <map_name> [options] > analysis_classic.log 2>&1 &
 
Example:
nohup python analysis_classic.py v3.1 baseline_division_of_labor > analysis_classic.log 2>&1 &
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from spoiled_broth.analysis.utils import MetricDefinitions

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
    
    # Add agent 2 metrics for classic experiments (since it's actually 2-agent)
    rewarded_metrics_2 = [
        "delivered_ai_rl_2",
        "cut_ai_rl_2",
        "salad_ai_rl_2",
    ]
    
    movement_metrics_2 = [
        "do_nothing_ai_rl_2",
        "floor_actions_ai_rl_2",
        "wall_actions_ai_rl_2",
        "useless_counter_actions_ai_rl_2",
        "useful_counter_actions_ai_rl_2",
        "useless_food_dispenser_actions_ai_rl_2",
        "useful_food_dispenser_actions_ai_rl_2",
        "useless_cutting_board_actions_ai_rl_2",
        "useful_cutting_board_actions_ai_rl_2",
        "useless_plate_dispenser_actions_ai_rl_2",
        "useful_plate_dispenser_actions_ai_rl_2",
        "useless_delivery_actions_ai_rl_2",
        "useful_delivery_actions_ai_rl_2",
    ]
    
    # Basic plots
    plotter.plot_basic_metrics(df, paths['figures_dir'], "total_deliveries", "Score (deliveries)")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_total", "Pure Reward")
    
    # Individual agent plots for both agents
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_ai_rl_2", "Pure Reward Agent 2")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "modified_reward_ai_rl_1", "Modified Reward Agent 1")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "modified_reward_ai_rl_2", "Modified Reward Agent 2")
    
    # Agent-specific metrics plots for both agents
    plotter.plot_agent_metrics(df, paths['figures_dir'], metrics['rewarded_metrics_1'], 1)
    plotter.plot_agent_metrics(df, paths['figures_dir'], rewarded_metrics_2, 2)
    
    # Smoothed plots
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "total_deliveries", "Score", by_attitude=False)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_total", "Pure Reward", by_attitude=False)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_ai_rl_2", "Pure Reward Agent 2")
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "modified_reward_ai_rl_1", "Modified Reward Agent 1")
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "modified_reward_ai_rl_2", "Modified Reward Agent 2")
    
    # Smoothed agent metrics for both agents - rewarded metrics
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['rewarded_metrics_1'], 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], rewarded_metrics_2, 2, smoothed=True)
    
    # Smoothed agent metrics for both agents - movement metrics
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['movement_metrics_1'], 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], movement_metrics_2, 2, smoothed=True)
    
    # Generate combined plots (use smoothing factor from config)
    smoothing_factor = analysis_results.get('config').smoothing_factor if analysis_results.get('config') else 15
    generate_combined_plots(df, paths, metrics['rewarded_metrics_1'], rewarded_metrics_2, 
                           metrics['movement_metrics_1'], movement_metrics_2, smoothing_factor)
    
    print(f"Classic analysis completed. Figures saved to {paths['figures_dir']}")


def generate_combined_plots(df, paths, rewarded_metrics_1, rewarded_metrics_2, 
                          movement_metrics_1, movement_metrics_2, smoothing_factor=15):
    """Generate combined plots showing both agents together."""

    
    N = smoothing_factor  # smoothing factor
    unique_attitudes = df["attitude_key"].unique()
    unique_game_type = df["game_type"].unique()
    unique_lr = df["lr"].unique()
    
    metric_labels = MetricDefinitions.get_metric_labels()
    metric_colors = MetricDefinitions.get_metric_colors()
    
    def get_metric_info(metric):
        """Get label and color for a metric."""
        base_metric = '_'.join(metric.split('_')[:-3])
        label = metric_labels.get(base_metric, base_metric.replace('_', ' ').title())
        color = metric_colors.get(base_metric, "#000000")
        return label, color
    
    # Combined rewarded metrics plots by attitude
    for attitude in unique_attitudes:
        subset = df[df["attitude_key"] == attitude]
        att_parts = attitude.split('_')
        att1_title = f"{att_parts[0]}_{att_parts[1]}"
        att2_title = f"{att_parts[2]}_{att_parts[3]}"
        
        for game_type in unique_game_type:
            for lr in unique_lr:
                filtered_subset = subset[(subset["game_type"] == game_type) & (subset["lr"] == lr)]
                
                if len(filtered_subset) == 0:
                    continue
                    
                filtered_subset = filtered_subset.copy()
                filtered_subset["epoch_block"] = (filtered_subset["epoch"] // N)
                
                # Create combined plot for rewarded metrics
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Agent 1 metrics
                for metric in rewarded_metrics_1:
                    label, color = get_metric_info(metric)
                    block_means = filtered_subset.groupby("epoch_block")[metric].mean()
                    middle_epochs = filtered_subset.groupby("epoch_block")["epoch"].median()
                    ax1.plot(middle_epochs, block_means, label=label, color=color)
                
                ax1.set_title(f"Agent 1 - Attitude {att1_title}")
                ax1.set_ylabel("Number of times the action was taken", fontsize=18)
                ax1.legend(fontsize=20, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
                
                # Agent 2 metrics
                for metric in rewarded_metrics_2:
                    label, color = get_metric_info(metric)
                    block_means = filtered_subset.groupby("epoch_block")[metric].mean()
                    middle_epochs = filtered_subset.groupby("epoch_block")["epoch"].median()
                    ax2.plot(middle_epochs, block_means, label=label, color=color)
                
                ax2.set_title(f"Agent 2 - Attitude {att2_title}")
                ax2.set_xlabel("Episodes", fontsize=20)
                ax2.set_ylabel("Number of times the action was taken", fontsize=18)
                ax2.legend(fontsize=20, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
                
                plt.tight_layout()
                
                sanitized_attitude = attitude.replace('.', 'p')
                filename_combined = f"rewarded_metrics_combined_avg_g{game_type}_lr{str(lr).replace('.', 'p')}_attitude_{sanitized_attitude}_smoothed_{N}.png"
                filepath_combined = os.path.join(paths['smoothed_figures_dir'], filename_combined)
                plt.savefig(filepath_combined)
                plt.close()
                
                # Create combined plot for movement metrics
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Agent 1 metrics
                for metric in movement_metrics_1:
                    label, color = get_metric_info(metric)
                    block_means = filtered_subset.groupby("epoch_block")[metric].mean()
                    middle_epochs = filtered_subset.groupby("epoch_block")["epoch"].median()
                    ax1.plot(middle_epochs, block_means, label=label, color=color)
                
                ax1.set_title(f"Agent 1 - Attitude {att1_title}")
                ax1.set_ylabel("Number of times the action was taken", fontsize=18)
                ax1.legend(
                    fontsize=16,
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='black'
                )
                
                # Agent 2 metrics
                for metric in movement_metrics_2:
                    label, color = get_metric_info(metric)
                    block_means = filtered_subset.groupby("epoch_block")[metric].mean()
                    middle_epochs = filtered_subset.groupby("epoch_block")["epoch"].median()
                    ax2.plot(middle_epochs, block_means, label=label, color=color)
                
                ax2.set_title(f"Agent 2 - Attitude {att2_title}")
                ax2.set_xlabel("Episodes", fontsize=20)
                ax2.set_ylabel("Number of times the action was taken", fontsize=18)
                ax2.legend(
                    fontsize=16,
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='black'
                )
                
                plt.tight_layout(rect=[0, 0, 0.99, 1])
                
                filename_combined = f"action_types_combined_avg_g{game_type}_lr{str(lr).replace('.', 'p')}_attitude_{sanitized_attitude}_smoothed_{N}.png"
                filepath_combined = os.path.join(paths['smoothed_figures_dir'], filename_combined)
                plt.savefig(filepath_combined)
                plt.close()
    
    print("Combined plots generated successfully.")


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