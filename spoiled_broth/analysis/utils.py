"""
Utility functions for analyzing training results from reinforcement learning experiments.

This module provides common functionality for:
- Data loading and preprocessing
- Configuration parsing
- Directory management
- Plotting utilities
- Statistical analysis

Author: Samuel Lozano
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from io import StringIO


class AnalysisConfig:
    """Configuration class for analysis parameters."""
    
    def __init__(self):
        # Graph settings
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        
        # Default smoothing factor
        self.smoothing_factor = 15
        
        # Cluster configurations
        self.cluster_paths = {
            'brigit': '/mnt/lustre/home/samuloza',
            'cuenca': '',
            'local': 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
        }


class DataProcessor:
    """Handles data loading, preprocessing and CSV operations."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.reward_pattern = re.compile(
            r"'([^']+)':\s*\(\s*([-\d\.eE+]+),\s*([-\d\.eE+]+)\)"
        )
    
    def setup_directories(self, experiment_type: str, intent_version: str, 
                         map_name: str, cluster: str = 'cuenca') -> Dict[str, str]:
        """
        Set up directory structure for analysis.
        
        Args:
            experiment_type: Type of experiment ('classic', 'competition', 'pretrained')
            intent_version: Version identifier
            map_name: Map name identifier  
            cluster: Cluster type ('brigit', 'cuenca', 'local')
            
        Returns:
            Dictionary containing all relevant paths
        """
        if cluster not in self.config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(self.config.cluster_paths.keys())}")
        
        local_path = self.config.cluster_paths[cluster]
        
        if intent_version:
            raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{intent_version}/map_{map_name}"
        else:
            raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/map_{map_name}"
        
        paths = {
            'raw_dir': raw_dir,
            'competitive_dir': f"{raw_dir}/competitive",
            'cooperative_dir': f"{raw_dir}/cooperative",
            'output_path': f"{raw_dir}/training_results.csv",
            'figures_dir': f"{raw_dir}/training_figures/",
            'smoothed_figures_dir': f"{raw_dir}/figures/smoothed_{self.config.smoothing_factor}/"
        }
        
        # Create directories
        base_dirs = [paths['competitive_dir'], paths['cooperative_dir']]
        for dir_path in base_dirs + [paths['figures_dir'], paths['smoothed_figures_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        return paths
    
    def parse_training_folder(self, folder_path: str, game_flag: int, 
                            num_agents: int = 1) -> Optional[pd.DataFrame]:
        """
        Parse a single training folder and extract data.
        
        Args:
            folder_path: Path to training folder
            game_flag: Game type flag (0=competitive, 1=cooperative)
            num_agents: Number of agents (1 for classic, 2 for competition)
            
        Returns:
            DataFrame with processed data or None if parsing fails
        """
        config_path = os.path.join(folder_path, "config.txt")
        csv_path = os.path.join(folder_path, "training_stats.csv")
        
        if not (os.path.exists(config_path) and os.path.exists(csv_path)):
            print(f"Missing config or CSV in {folder_path}")
            return None
        
        # Parse config file
        with open(config_path, "r") as f:
            config_contents = f.read()
        
        matches = self.reward_pattern.findall(config_contents)
        if len(matches) != num_agents:
            print(f"Expected {num_agents} agents, found {len(matches)} in {folder_path}")
            return None
        
        # Extract learning rate
        lr_match = re.search(r"LR:\s*([0-9.eE+-]+)", config_contents)
        if not lr_match:
            print(f"Learning rate not found in {folder_path}")
            return None
        lr = float(lr_match.group(1))
        
        # Load and clean CSV data
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        header = lines[0]
        filtered_lines = [header] + [line for line in lines[1:] if not line.startswith("episode,env")]
        
        df = pd.read_csv(StringIO("".join(filtered_lines)))
        df.iloc[:, 0] = range(1, len(df) + 1)
        
        # Extract folder timestamp
        folder_name = os.path.basename(folder_path)
        date_time_str = folder_name.replace("Training_", "")
        
        # Add metadata columns
        df.insert(0, "timestamp", date_time_str)
        df.insert(1, "game_type", game_flag)
        
        # Add agent parameters
        for i, (name, alpha, beta) in enumerate(matches, 1):
            df.insert(1 + i, f"alpha_{i}", float(alpha))
            df.insert(2 + i, f"beta_{i}", float(beta))
        
        df.insert(len(matches) * 2 + 2, "lr", lr)
        
        return df
    
    def load_experiment_data(self, paths: Dict[str, str], 
                           num_agents: int = 1) -> pd.DataFrame:
        """
        Load all experiment data from competitive and cooperative directories.
        
        Args:
            paths: Dictionary containing directory paths
            num_agents: Number of agents in the experiment
            
        Returns:
            Combined DataFrame with all experiment data
        """
        all_dfs = []
        
        base_dirs = {
            "Competitive": paths['competitive_dir'],
            "Cooperative": paths['cooperative_dir']
        }
        print("Loading experiment data from directories:", base_dirs.items())

        for game_type, base_dir in base_dirs.items():
            if not os.path.exists(base_dir):
                continue
                
            game_flag = 1 if "Cooperative" in game_type else 0
            
            for folder in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                
                df = self.parse_training_folder(folder_path, game_flag, num_agents)
                if df is not None:
                    all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No valid training data found")
        
        # Combine all dataframes
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove existing output file and save new one
        if os.path.exists(paths['output_path']):
            os.remove(paths['output_path'])
        
        final_df.to_csv(paths['output_path'], index=False)
        return final_df
    
    def prepare_dataframe(self, df: pd.DataFrame, num_agents: int = 1) -> pd.DataFrame:
        """
        Prepare dataframe with computed columns and proper types.
        
        Args:
            df: Raw dataframe
            num_agents: Number of agents
            
        Returns:
            Processed dataframe
        """
        # Set data types
        dtype_dict = {
            "timestamp": str,
            "game_type": int,
        }
        
        for i in range(1, num_agents + 1):
            dtype_dict[f"alpha_{i}"] = float
            dtype_dict[f"beta_{i}"] = float
        
        # Convert numeric columns
        for col in df.columns[6:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by alpha values
        alpha_cols = [f"alpha_{i}" for i in range(1, num_agents + 1)]
        df = df.sort_values(by=alpha_cols, ascending=[False] * len(alpha_cols))
        
        # Create attitude key
        if num_agents == 1:
            df["attitude_key"] = df.apply(lambda row: f"{row['alpha_1']}_{row['beta_1']}", axis=1)
            df["pure_reward_total"] = df["pure_reward_ai_rl_1"]
            df["total_deliveries"] = np.where(
                df["game_type"] == 1,
                df["delivered_ai_rl_1"],
                df["delivered_ai_rl_1"]
            )
        else:  # num_agents == 2
            df["attitude_key"] = df.apply(
                lambda row: f"{row['alpha_1']}_{row['beta_1']}_{row['alpha_2']}_{row['beta_2']}", 
                axis=1
            )
            df["pure_reward_total"] = df["pure_reward_ai_rl_1"] + df["pure_reward_ai_rl_2"]
            # For classic experiments, total deliveries is always the sum of both agents
            df["total_deliveries"] = df["delivered_ai_rl_1"] + df["delivered_ai_rl_2"]
        
        return df


class MetricDefinitions:
    """Defines metrics and their visual properties for different experiment types."""
    
    @staticmethod
    def get_classic_metrics() -> Dict:
        """Get metric definitions for classic experiments."""
        return {
            'rewarded_metrics_1': [
                "delivered_ai_rl_1",
                "cut_ai_rl_1", 
                "salad_ai_rl_1",
            ],
            'movement_metrics_1': [
                "do_nothing_ai_rl_1",
                "floor_actions_ai_rl_1",
                "wall_actions_ai_rl_1",
                "useless_counter_actions_ai_rl_1",
                "useful_counter_actions_ai_rl_1",
                "useless_food_dispenser_actions_ai_rl_1",
                "useful_food_dispenser_actions_ai_rl_1",
                "useless_cutting_board_actions_ai_rl_1",
                "useful_cutting_board_actions_ai_rl_1",
                "useless_plate_dispenser_actions_ai_rl_1",
                "useful_plate_dispenser_actions_ai_rl_1",
                "useless_delivery_actions_ai_rl_1",
                "useful_delivery_actions_ai_rl_1",
            ]
        }
    
    @staticmethod
    def get_competition_metrics() -> Dict:
        """Get metric definitions for competition experiments."""
        return {
            'result_events_1': [
                "delivered_own_ai_rl_1",
                "delivered_other_ai_rl_1",
                "salad_own_ai_rl_1",
                "salad_other_ai_rl_1",
                "cut_own_ai_rl_1",
                "cut_other_ai_rl_1",
            ],
            'result_events_2': [
                "delivered_own_ai_rl_2",
                "delivered_other_ai_rl_2",
                "salad_own_ai_rl_2",
                "salad_other_ai_rl_2",
                "cut_own_ai_rl_2",
                "cut_other_ai_rl_2",
            ],
            'action_types_1': [
                "do_nothing_ai_rl_1",
                "floor_actions_ai_rl_1",
                "wall_actions_ai_rl_1",
                "useless_counter_actions_ai_rl_1",
                "useful_counter_actions_ai_rl_1",
                "useless_own_food_dispenser_actions_ai_rl_1",
                "useful_own_food_dispenser_actions_ai_rl_1",
                "useless_other_food_dispenser_actions_ai_rl_1",
                "useful_other_food_dispenser_actions_ai_rl_1",
                "useless_cutting_board_actions_ai_rl_1",
                "useful_own_cutting_board_actions_ai_rl_1",
                "useful_other_cutting_board_actions_ai_rl_1",
                "useless_plate_dispenser_actions_ai_rl_1",
                "useful_plate_dispenser_actions_ai_rl_1",
                "useless_delivery_actions_ai_rl_1",
                "useful_own_delivery_actions_ai_rl_1",
                "useful_other_delivery_actions_ai_rl_1",
            ],
            'action_types_2': [
                "do_nothing_ai_rl_2",
                "floor_actions_ai_rl_2", 
                "wall_actions_ai_rl_2",
                "useless_counter_actions_ai_rl_2",
                "useful_counter_actions_ai_rl_2",
                "useless_own_food_dispenser_actions_ai_rl_2",
                "useful_own_food_dispenser_actions_ai_rl_2",
                "useless_other_food_dispenser_actions_ai_rl_2",
                "useful_other_food_dispenser_actions_ai_rl_2",
                "useless_cutting_board_actions_ai_rl_2",
                "useful_own_cutting_board_actions_ai_rl_2",
                "useful_other_cutting_board_actions_ai_rl_2",
                "useless_plate_dispenser_actions_ai_rl_2",
                "useful_plate_dispenser_actions_ai_rl_2",
                "useless_delivery_actions_ai_rl_2",
                "useful_own_delivery_actions_ai_rl_2",
                "useful_other_delivery_actions_ai_rl_2",
            ]
        }
    
    @staticmethod
    def get_metric_labels() -> Dict:
        """Get human-readable labels for metrics."""
        return {
            # Classic metrics
            "delivered": "Delivered",
            "cut": "Cut", 
            "salad": "Salad",
            "do_nothing": "No action",
            "floor_actions": "Floor",
            "wall_actions": "Wall",
            "useless_counter_actions": "Useless Counter",
            "useful_counter_actions": "Useful Counter",
            "useless_food_dispenser_actions": "Useless Food Dispenser",
            "useful_food_dispenser_actions": "Useful Food Dispenser",
            "useless_cutting_board_actions": "Useless Cutting Board",
            "useful_cutting_board_actions": "Useful Cutting Board",
            "useless_plate_dispenser_actions": "Useless Plate Dispenser",
            "useful_plate_dispenser_actions": "Useful Plate Dispenser",
            "useless_delivery_actions": "Useless Delivery",
            "useful_delivery_actions": "Useful Delivery",
            
            # Competition metrics
            "delivered_own": "Delivered Own",
            "delivered_other": "Delivered Other",
            "salad_own": "Salad Own",
            "salad_other": "Salad Other",
            "cut_own": "Cut Own",
            "cut_other": "Cut Other",
            "useless_own_food_dispenser_actions": "Useless Own Food Dispenser",
            "useful_own_food_dispenser_actions": "Useful Own Food Dispenser",
            "useless_other_food_dispenser_actions": "Useless Other Food Dispenser",
            "useful_other_food_dispenser_actions": "Useful Other Food Dispenser",
            "useful_own_cutting_board_actions": "Useful Own Cutting Board",
            "useful_other_cutting_board_actions": "Useful Other Cutting Board",
            "useful_own_delivery_actions": "Useful Own Delivery",
            "useful_other_delivery_actions": "Useful Other Delivery",
        }
    
    @staticmethod
    def get_metric_colors() -> Dict:
        """Get color scheme for metrics."""
        return {
            # Classic colors
            "delivered": "#27AE60",
            "cut": "#2980B9",
            "salad": "#E67E22",
            "do_nothing": "#000000",
            "floor_actions": "#9B59B6",
            "wall_actions": "#59351F",
            "useless_counter_actions": "#D5D8DC",
            "useful_counter_actions": "#7B7D7D",
            "useless_food_dispenser_actions": "#F5B7B1",
            "useful_food_dispenser_actions": "#C0392B",
            "useless_cutting_board_actions": "#AED6F1",
            "useful_cutting_board_actions": "#2980B9",
            "useless_plate_dispenser_actions": "#FAD7A0",
            "useful_plate_dispenser_actions": "#E67E22",
            "useless_delivery_actions": "#A9DFBF",
            "useful_delivery_actions": "#27AE60",
            
            # Competition colors
            "delivered_own": "#A9DFBF",
            "delivered_other": "#27AE60",
            "salad_own": "#F5CBA7",
            "salad_other": "#E67E22",
            "cut_own": "#AED6F1",
            "cut_other": "#2980B9",
            "useless_own_food_dispenser_actions": "#F5B7B1",
            "useful_own_food_dispenser_actions": "#C0392B",
            "useless_other_food_dispenser_actions": "#D7BDE2",
            "useful_other_food_dispenser_actions": "#8E44AD",
            "useful_own_cutting_board_actions": "#2980B9",
            "useful_other_cutting_board_actions": "#1ABC9C",
            "useful_own_delivery_actions": "#27AE60",
            "useful_other_delivery_actions": "#145A32",
        }


class PlotGenerator:
    """Generates various types of plots for analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.metric_labels = MetricDefinitions.get_metric_labels()
        self.metric_colors = MetricDefinitions.get_metric_colors()
    
    def _sanitize_filename(self, attitude: str) -> str:
        """Sanitize attitude string for filename use."""
        return attitude.replace('.', 'p')
    
    def _get_metric_info(self, metric: str) -> Tuple[str, str]:
        """Get label and color for a metric."""
        # Extract base metric name
        base_metric = '_'.join(metric.split('_')[:-3])  # Remove _ai_rl_X suffix
        
        label = self.metric_labels.get(base_metric, base_metric.replace('_', ' ').title())
        color = self.metric_colors.get(base_metric, "#000000")
        
        return label, color
    
    def plot_basic_metrics(self, df: pd.DataFrame, figures_dir: str,
                          metric_col: str, metric_name: str, by_attitude: bool = True):
        """
        Generate basic metric plots (score, reward, etc.).
        
        Args:
            df: Data DataFrame
            figures_dir: Directory to save figures
            metric_col: Column name for the metric
            metric_name: Human-readable metric name
            by_attitude: Whether to plot by attitude or overall
        """
        unique_attitudes = df["attitude_key"].unique()
        unique_lr = df["lr"].unique()
        unique_game_type = df["game_type"].unique()
        
        if by_attitude:
            for attitude in unique_attitudes:
                subset = df[df["attitude_key"] == attitude]
                
                plt.figure(figsize=(10, 6))
                
                for game_type in unique_game_type:
                    game_type_filtered = subset[subset["game_type"] == game_type]
                    
                    for lr in unique_lr:
                        lr_filtered = game_type_filtered[game_type_filtered["lr"] == lr]
                        grouped = lr_filtered.groupby("epoch")[metric_col].mean().reset_index()
                        label = f"Game Type {game_type}, LR {lr}"
                        plt.plot(grouped["epoch"], grouped[metric_col], label=label)
                
                plt.title(f"{metric_name} vs Epoch\nAttitude {attitude}")
                plt.xlabel("Epoch")
                plt.ylabel(metric_name)
                plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = self._sanitize_filename(attitude)
                filename = f"{metric_name.lower().replace(' ', '_')}_attitude_{sanitized_attitude}.png"
                plt.savefig(os.path.join(figures_dir, filename))
                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            
            for game_type in unique_game_type:
                game_type_filtered = df[df["game_type"] == game_type]
                
                for lr in unique_lr:
                    lr_filtered = game_type_filtered[game_type_filtered["lr"] == lr]
                    grouped = lr_filtered.groupby("epoch")[metric_col].mean().reset_index()
                    
                    if game_type == 0:
                        label = "Competitive"
                        color = "red"
                    else:
                        label = "Cooperative"
                        color = "blue"
                    
                    plt.plot(grouped["epoch"], grouped[metric_col], label=label, color=color)
            
            plt.xlabel("Epochs", fontsize=20)
            plt.ylabel(f"Mean {metric_name.lower()}", fontsize=20)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            
            filename = f"{metric_name.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(figures_dir, filename))
            plt.close()
    
    def plot_smoothed_metrics(self, df: pd.DataFrame, figures_dir: str,
                             metric_col: str, metric_name: str, by_attitude: bool = True):
        """Generate smoothed versions of metric plots."""
        N = self.config.smoothing_factor
        unique_attitudes = df["attitude_key"].unique()
        unique_lr = df["lr"].unique()
        unique_game_type = df["game_type"].unique()
        
        if by_attitude:
            for attitude in unique_attitudes:
                subset = df[df["attitude_key"] == attitude]
                
                plt.figure(figsize=(10, 6))
                
                for game_type in unique_game_type:
                    game_type_filtered = subset[subset["game_type"] == game_type]
                    
                    for lr in unique_lr:
                        lr_filtered = game_type_filtered[game_type_filtered["lr"] == lr]
                        lr_filtered = lr_filtered.copy()
                        lr_filtered["epoch_block"] = (lr_filtered["epoch"] // N)
                        block_means = lr_filtered.groupby("epoch_block")[metric_col].mean()
                        middle_epochs = lr_filtered.groupby("epoch_block")["epoch"].median()
                        
                        label = f"Game Type {game_type}, LR {lr}"
                        plt.plot(middle_epochs, block_means, label=label)
                
                plt.title(f"{metric_name} vs Epoch\nAttitude {attitude}")
                plt.xlabel("Epoch")
                plt.ylabel(metric_name)
                plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = self._sanitize_filename(attitude)
                filename = f"{metric_name.lower().replace(' ', '_')}_attitude_{sanitized_attitude}_smoothed_{N}.png"
                plt.savefig(os.path.join(figures_dir, filename))
                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            
            for game_type in unique_game_type:
                game_type_filtered = df[df["game_type"] == game_type]
                
                for lr in unique_lr:
                    lr_filtered = game_type_filtered[game_type_filtered["lr"] == lr]
                    lr_filtered = lr_filtered.copy()
                    lr_filtered["epoch_block"] = (lr_filtered["epoch"] // N)
                    block_means = lr_filtered.groupby("epoch_block")[metric_col].mean()
                    middle_epochs = lr_filtered.groupby("epoch_block")["epoch"].median()
                    
                    if game_type == 0:
                        label = "Competitive"
                        color = "red"
                    else:
                        label = "Cooperative" 
                        color = "blue"
                    
                    plt.plot(middle_epochs, block_means, label=label, color=color)
            
            plt.xlabel("Epochs", fontsize=20)
            plt.ylabel(f"Mean {metric_name.lower()}", fontsize=20)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            
            filename = f"{metric_name.lower().replace(' ', '_')}_smoothed_{N}.png"
            plt.savefig(os.path.join(figures_dir, filename))
            plt.close()
    
    def plot_agent_metrics(self, df: pd.DataFrame, figures_dir: str, metrics: List[str],
                          agent_num: int, smoothed: bool = False):
        """Plot individual agent metrics."""
        N = self.config.smoothing_factor if smoothed else 1
        unique_game_type = df["game_type"].unique()
        unique_lr = df["lr"].unique()
        
        for game_type in unique_game_type:
            for lr in unique_lr:
                filtered_subset = df[(df["game_type"] == game_type) & (df["lr"] == lr)]
                
                if smoothed:
                    filtered_subset = filtered_subset.copy()
                    filtered_subset["epoch_block"] = (filtered_subset["epoch"] // N)
                
                plt.figure(figsize=(12, 6))
                
                for metric in metrics:
                    label, color = self._get_metric_info(metric)
                    
                    if smoothed:
                        block_means = filtered_subset.groupby("epoch_block")[metric].mean()
                        middle_epochs = filtered_subset.groupby("epoch_block")["epoch"].median()
                        plt.plot(middle_epochs, block_means, label=label, color=color)
                    else:
                        grouped = filtered_subset.groupby("epoch")[metric].mean().reset_index()
                        plt.plot(grouped["epoch"], grouped[metric], label=label, color=color)
                
                title = f"Metrics per Epoch - Game Type {game_type}, LR {lr}"
                if smoothed:
                    title += f" (Smoothed {N})"
                
                plt.title(title)
                plt.xlabel("Epoch")
                plt.ylabel("Mean value")
                plt.legend()
                plt.tight_layout()
                
                suffix = f"_smoothed_{N}" if smoothed else ""
                filename = f"metrics_agent{agent_num}_g{game_type}_lr{str(lr).replace('.', 'p')}{suffix}.png"
                plt.savefig(os.path.join(figures_dir, filename))
                plt.close()


def setup_argument_parser(experiment_type: str) -> argparse.ArgumentParser:
    """
    Set up command line argument parser.
    
    Args:
        experiment_type: Type of experiment ('classic', 'competition', 'pretrained')
        
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=f'Analysis script for {experiment_type} experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'intent_version',
        type=str,
        help='Intent version identifier (e.g., v3.1)'
    )
    
    parser.add_argument(
        'map_name',
        type=str,
        help='Map name identifier (e.g., simple_kitchen_circular)'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        choices=['brigit', 'cuenca', 'local'],
        default='cuenca',
        help='Cluster type for path configuration'
    )
    
    parser.add_argument(
        '--smoothing-factor',
        type=int,
        default=15,
        help='Smoothing factor for curve generation'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for figures'
    )
    
    return parser


def main_analysis_pipeline(experiment_type: str, intent_version: str, map_name: str,
                          cluster: str = 'cuenca', smoothing_factor: int = 15) -> Dict:
    """
    Main analysis pipeline that can be used by all experiment types.
    
    Args:
        experiment_type: Type of experiment
        intent_version: Version identifier
        map_name: Map name
        cluster: Cluster type
        smoothing_factor: Smoothing factor for plots
        
    Returns:
        Dictionary containing processed data and paths
    """
    # Initialize components
    config = AnalysisConfig()
    config.smoothing_factor = smoothing_factor
    
    processor = DataProcessor(config)
    plotter = PlotGenerator(config)
    
    # Set up directories and load data
    paths = processor.setup_directories(experiment_type, intent_version, map_name, cluster)
    
    # Determine number of agents based on experiment type
    # Classic experiments are actually 2-agent experiments
    num_agents = 2
    
    # Load and process data
    raw_df = processor.load_experiment_data(paths, num_agents)
    df = processor.prepare_dataframe(raw_df, num_agents)
    
    print(f"Loaded {len(df)} training records")
    print(f"Unique attitudes: {len(df['attitude_key'].unique())}")
    print(f"Figures will be saved to: {paths['figures_dir']}")
    
    return {
        'df': df,
        'paths': paths,
        'config': config,
        'processor': processor,
        'plotter': plotter,
        'num_agents': num_agents
    }