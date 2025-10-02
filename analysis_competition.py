#!/usr/bin/env python3
"""
Analysis script for competition reinforcement learning experiments.

This script analyzes training results from two-agent competitive experiments,
generating comprehensive visualizations and statistics.

Usage:
nohup python analysis_competition.py <intent_version> <map_name> [options] > analysis_competition.log 2>&1 &

Example:
nohup python analysis_competition.py v3.1 simple_kitchen_competition --cluster cuenca --smoothing-factor 15 > analysis_competition.log 2>&1 &
"""

import sys
import os
import numpy as np

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import (
    setup_argument_parser, main_analysis_pipeline, MetricDefinitions
)


def generate_competition_plots(analysis_results):
    """Generate all plots specific to competition experiments."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    plotter = analysis_results['plotter']
    config = analysis_results['config']
    
    print("Generating competition experiment plots...")
    
    # Get competition-specific metrics
    metrics = MetricDefinitions.get_competition_metrics()
    
    # Basic plots
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_total", "Pure Reward")
    
    # Individual agent plots
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_ai_rl_2", "Pure Reward Agent 2")
    
    # Agent-specific metrics plots
    plotter.plot_agent_metrics(df, paths['figures_dir'], metrics['result_events_1'], 1)
    plotter.plot_agent_metrics(df, paths['figures_dir'], metrics['result_events_2'], 2)
    
    # Smoothed plots
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "total_deliveries", "Total Deliveries", by_attitude=False)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_total", "Pure Reward", by_attitude=False)
    
    # Individual agent deliveries (smoothed)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "total_deliveries_ai_rl_1", "Total Deliveries Agent 1")
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "total_deliveries_ai_rl_2", "Total Deliveries Agent 2")
    
    # Pure rewards per agent (smoothed)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_ai_rl_2", "Pure Reward Agent 2")
    
    # Modified rewards per agent (smoothed)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "modified_reward_ai_rl_1", "Modified Reward Agent 1")
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "modified_reward_ai_rl_2", "Modified Reward Agent 2")
    
    # Smoothed agent metrics
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['result_events_1'], 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['result_events_2'], 2, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['action_types_1'], 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['action_types_2'], 2, smoothed=True)
    
    # Generate attitude-specific analysis
    generate_attitude_analysis(analysis_results)
    
    print(f"Competition analysis completed. Figures saved to {paths['figures_dir']}")


def generate_attitude_analysis(analysis_results):
    """Generate analysis plots grouped by individual attitudes."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results['config']
    
    print("Generating attitude-specific analysis...")
    
    # Create individual attitude keys for each agent
    df['attitude_agent_1'] = df['alpha_1'].astype(str) + '_' + df['beta_1'].astype(str)
    df['attitude_agent_2'] = df['alpha_2'].astype(str) + '_' + df['beta_2'].astype(str)
    
    # Get unique individual attitudes
    unique_attitudes = df["attitude_key"].unique()
    unique_individual_attitudes = set()
    
    for attitude in unique_attitudes:
        att_parts = attitude.split('_')
        unique_individual_attitudes.add(f"{att_parts[0]}_{att_parts[1]}")  # agent 1 attitude
        unique_individual_attitudes.add(f"{att_parts[2]}_{att_parts[3]}")  # agent 2 attitude
    
    print(f"Individual attitudes found: {sorted(unique_individual_attitudes)}")
    
    # Generate plots for individual attitudes
    generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config)
    
    # Generate combined attitude plots
    generate_combined_attitude_plots(df, paths, unique_attitudes, config)


def generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config):
    """Generate plots for individual attitudes regardless of partner."""
    import matplotlib.pyplot as plt
    
    N = config.smoothing_factor
    unique_lr = df["lr"].unique()
    unique_game_type = df["game_type"].unique()
    
    for individual_attitude in unique_individual_attitudes:
        att_parts = individual_attitude.split('_')
        alpha = float(att_parts[0])
        beta = float(att_parts[1])
        
        # Calculate degree for title
        if alpha == 0 and beta == 0:
            degree = 0
        else:
            degree = np.degrees(np.arctan2(beta, alpha)) % 360
        
        for game_type in unique_game_type:
            for lr in unique_lr:
                # Filter data where either agent has this attitude
                mask_agent_1 = (df['attitude_agent_1'] == individual_attitude)
                mask_agent_2 = (df['attitude_agent_2'] == individual_attitude) 
                mask_conditions = (df["game_type"] == game_type) & (df["lr"] == lr)
                
                filtered_subset = df[mask_conditions & (mask_agent_1 | mask_agent_2)].copy()
                
                if len(filtered_subset) > 0:
                    filtered_subset["epoch_block"] = (filtered_subset["epoch"] // N)
                    
                    # Plot delivery metrics
                    plt.figure(figsize=(12, 6))
                    
                    # Combine metrics from both agents when they have this attitude
                    delivery_own_combined = []
                    delivery_other_combined = []
                    middle_epochs = []
                    
                    for epoch_block in filtered_subset["epoch_block"].unique():
                        block_data = filtered_subset[filtered_subset["epoch_block"] == epoch_block]
                        
                        own_total = 0
                        other_total = 0
                        count = 0
                        
                        for _, row in block_data.iterrows():
                            if row['attitude_agent_1'] == individual_attitude:
                                own_total += row['delivered_own_ai_rl_1']
                                other_total += row['delivered_other_ai_rl_1']
                                count += 1
                            if row['attitude_agent_2'] == individual_attitude:
                                own_total += row['delivered_own_ai_rl_2'] 
                                other_total += row['delivered_other_ai_rl_2']
                                count += 1
                        
                        if count > 0:
                            delivery_own_combined.append(own_total / count)
                            delivery_other_combined.append(other_total / count)
                            middle_epochs.append(block_data["epoch"].median())
                    
                    plt.plot(middle_epochs, delivery_own_combined, label="Delivered Own", color="#A9DFBF")
                    plt.plot(middle_epochs, delivery_other_combined, label="Delivered Other", color="#27AE60")
                    
                    plt.title(f"Delivery Metrics - Individual Attitude {individual_attitude} ({degree:.1f}°)\n"
                             f"Game Type {game_type}, LR {lr} (Smoothed {N})")
                    plt.xlabel("Epoch")
                    plt.ylabel("Mean value")
                    plt.legend()
                    plt.tight_layout()
                    
                    sanitized_attitude = individual_attitude.replace('.', 'p')
                    filename = f"delivery_individual_attitude_{sanitized_attitude}_g{game_type}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
                    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                    plt.close()


def generate_combined_attitude_plots(df, paths, unique_attitudes, config):
    """Generate plots showing both agents together for each attitude combination."""
    import matplotlib.pyplot as plt
    
    N = config.smoothing_factor
    unique_game_type = df["game_type"].unique()
    
    for attitude in unique_attitudes:
        subset = df[df["attitude_key"] == attitude]
        att_parts = attitude.split('_')
        att1_title = f"{att_parts[0]}_{att_parts[1]}" 
        att2_title = f"{att_parts[2]}_{att_parts[3]}"
        
        for game_type in unique_game_type:
            game_filtered = subset[subset["game_type"] == game_type]
            
            if len(game_filtered) > 0:
                game_filtered = game_filtered.copy()
                game_filtered["epoch_block"] = (game_filtered["epoch"] // N)
                
                # Plot combined delivery metrics
                plt.figure(figsize=(12, 6))
                
                # Agent 1 metrics
                block_means_own_1 = game_filtered.groupby("epoch_block")["delivered_own_ai_rl_1"].mean()
                block_means_other_1 = game_filtered.groupby("epoch_block")["delivered_other_ai_rl_1"].mean()
                middle_epochs = game_filtered.groupby("epoch_block")["epoch"].median()
                
                plt.plot(middle_epochs, block_means_own_1, label=f"Agent 1 Own ({att1_title})", 
                        color="#A9DFBF", linestyle='-')
                plt.plot(middle_epochs, block_means_other_1, label=f"Agent 1 Other ({att1_title})", 
                        color="#27AE60", linestyle='-')
                
                # Agent 2 metrics
                block_means_own_2 = game_filtered.groupby("epoch_block")["delivered_own_ai_rl_2"].mean()
                block_means_other_2 = game_filtered.groupby("epoch_block")["delivered_other_ai_rl_2"].mean()
                
                plt.plot(middle_epochs, block_means_own_2, label=f"Agent 2 Own ({att2_title})", 
                        color="#AED6F1", linestyle='--')
                plt.plot(middle_epochs, block_means_other_2, label=f"Agent 2 Other ({att2_title})", 
                        color="#2980B9", linestyle='--')
                
                plt.title(f"Combined Delivery Metrics - Attitude {attitude}\n"
                         f"Game Type {game_type} (Smoothed {N})")
                plt.xlabel("Epoch")
                plt.ylabel("Mean deliveries")
                plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = attitude.replace('.', 'p')
                filename = f"combined_delivery_attitude_{sanitized_attitude}_g{game_type}_smoothed_{N}.png"
                plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                plt.close()


def main():
    """Main execution function."""
    parser = setup_argument_parser('competition')
    args = parser.parse_args()
    
    print(f"Starting competition experiment analysis...")
    print(f"Intent version: {args.intent_version}")
    print(f"Map: {args.map_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Smoothing factor: {args.smoothing_factor}")
    
    try:
        # Run main analysis pipeline
        analysis_results = main_analysis_pipeline(
            experiment_type='competition',
            intent_version=args.intent_version,
            map_name=args.map_name,
            cluster=args.cluster,
            smoothing_factor=args.smoothing_factor
        )
        
        # Generate competition-specific plots
        generate_competition_plots(analysis_results)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()