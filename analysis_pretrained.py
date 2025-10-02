#!/usr/bin/env python3
"""
Analysis script for pretrained reinforcement learning experiments.

This script analyzes training results from experiments using pretrained models,
generating comprehensive visualizations and statistics.

Usage:
nohup python analysis_pretrained.py <intent_version> <map_name> [options] > analysis_pretrained.log 2>&1 &

Example:
nohup python analysis_pretrained.py v3.1 simple_kitchen_circular --cluster cuenca --smoothing-factor 15 > analysis_pretrained.log 2>&1 &
"""

import sys
import os
import numpy as np

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import (
    setup_argument_parser, main_analysis_pipeline, MetricDefinitions
)


def generate_pretrained_plots(analysis_results):
    """Generate all plots specific to pretrained experiments."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    plotter = analysis_results['plotter']
    
    print("Generating pretrained experiment plots...")
    
    # Get classic-specific metrics (pretrained uses similar structure to classic)
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
    
    # Generate attitude-specific analysis
    generate_attitude_analysis(analysis_results)
    
    print(f"Pretrained analysis completed. Figures saved to {paths['figures_dir']}")


def generate_attitude_analysis(analysis_results):
    """Generate analysis plots grouped by individual attitudes."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results['config']
    
    print("Generating attitude-specific analysis...")
    
    # Create individual attitude keys for each agent
    df['attitude_agent_1'] = df['alpha_1'].astype(str) + '_' + df['beta_1'].astype(str)
    
    # Get unique individual attitudes
    unique_attitudes = df["attitude_key"].unique()
    unique_individual_attitudes = set()
    
    for attitude in unique_attitudes:
        att_parts = attitude.split('_')
        unique_individual_attitudes.add(f"{att_parts[0]}_{att_parts[1]}")  # agent 1 attitude
    
    print(f"Individual attitudes found: {sorted(unique_individual_attitudes)}")
    
    # Generate plots for individual attitudes
    generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config)
    
    # Generate combined attitude plots
    generate_combined_attitude_plots(df, paths, unique_attitudes, config)


def generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config):
    """Generate plots for individual attitudes with averaged metrics."""
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
                # Filter data where agent has this attitude
                mask_agent_1 = (df['attitude_agent_1'] == individual_attitude)
                mask_conditions = (df["game_type"] == game_type) & (df["lr"] == lr)
                
                filtered_subset = df[mask_conditions & mask_agent_1].copy()
                
                if len(filtered_subset) > 0:
                    filtered_subset["epoch_block"] = (filtered_subset["epoch"] // N)
                    
                    # Plot rewarded metrics
                    plt.figure(figsize=(12, 6))
                    
                    rewarded_metrics = ["delivered_ai_rl_1", "cut_ai_rl_1", "salad_ai_rl_1"]
                    colors = ["#27AE60", "#2980B9", "#E67E22"]
                    labels = ["Delivered", "Cut", "Salad"]
                    
                    for metric, color, label in zip(rewarded_metrics, colors, labels):
                        if metric in filtered_subset.columns:
                            block_means = filtered_subset.groupby("epoch_block")[metric].mean()
                            middle_epochs = filtered_subset.groupby("epoch_block")["epoch"].median()
                            plt.plot(middle_epochs, block_means, label=label, color=color)
                    
                    plt.title(f"Rewarded Metrics - Individual Attitude {individual_attitude} ({degree:.1f}°)\n"
                             f"Game Type {game_type}, LR {lr} (Smoothed {N})")
                    plt.xlabel("Epoch")
                    plt.ylabel("Mean value")
                    plt.legend()
                    plt.tight_layout()
                    
                    sanitized_attitude = individual_attitude.replace('.', 'p')
                    filename = f"rewarded_individual_attitude_{sanitized_attitude}_g{game_type}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
                    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                    plt.close()


def generate_combined_attitude_plots(df, paths, unique_attitudes, config):
    """Generate plots showing metrics averaged over all other agent attitudes."""
    import matplotlib.pyplot as plt
    
    N = config.smoothing_factor
    unique_game_type = df["game_type"].unique()
    unique_lr = df["lr"].unique()
    
    for attitude in unique_attitudes:
        subset = df[df["attitude_key"] == attitude]
        att_parts = attitude.split('_')
        att1_title = f"{att_parts[0]}_{att_parts[1]}"
        
        for game_type in unique_game_type:
            for lr in unique_lr:
                game_lr_filtered = subset[(subset["game_type"] == game_type) & (subset["lr"] == lr)]
                
                if len(game_lr_filtered) > 0:
                    game_lr_filtered = game_lr_filtered.copy()
                    game_lr_filtered["epoch_block"] = (game_lr_filtered["epoch"] // N)
                    
                    # Plot rewarded metrics averaged
                    plt.figure(figsize=(12, 6))
                    
                    rewarded_metrics = ["delivered_ai_rl_1", "cut_ai_rl_1", "salad_ai_rl_1"]
                    colors = ["#27AE60", "#2980B9", "#E67E22"]
                    labels = ["Delivered", "Cut", "Salad"]
                    
                    for metric, color, label in zip(rewarded_metrics, colors, labels):
                        if metric in game_lr_filtered.columns:
                            block_means = game_lr_filtered.groupby("epoch_block")[metric].mean()
                            middle_epochs = game_lr_filtered.groupby("epoch_block")["epoch"].median()
                            plt.plot(middle_epochs, block_means, label=label, color=color)
                    
                    plt.title(f"Rewarded Metrics (Averaged) - Attitude {att1_title}\n"
                             f"Game Type {game_type}, LR {lr} (Smoothed {N})")
                    plt.xlabel("Epoch")
                    plt.ylabel("Mean value")
                    plt.legend()
                    plt.tight_layout()
                    
                    sanitized_attitude = attitude.replace('.', 'p')
                    filename = f"rewarded_avg_attitude_{sanitized_attitude}_g{game_type}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
                    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                    plt.close()


def main():
    """Main execution function."""
    parser = setup_argument_parser('pretrained')
    args = parser.parse_args()
    
    print(f"Starting pretrained experiment analysis...")
    print(f"Intent version: {args.intent_version}")
    print(f"Map: {args.map_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Smoothing factor: {args.smoothing_factor}")
    
    try:
        # Run main analysis pipeline
        analysis_results = main_analysis_pipeline(
            experiment_type='pretraining',  # Note: using 'pretraining' to match directory structure
            intent_version=args.intent_version,
            map_name=args.map_name,
            cluster=args.cluster,
            smoothing_factor=args.smoothing_factor
        )
        
        # Generate pretrained-specific plots
        generate_pretrained_plots(analysis_results)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()