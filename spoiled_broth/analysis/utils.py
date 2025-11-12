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
    
    def setup_directories(self, experiment_type: str, map_name: str, cluster: str = 'cuenca') -> Dict[str, str]:
        """
        Set up directory structure for analysis.
        
        Args:
            experiment_type: Type of experiment ('classic', 'competition', 'pretrained')
            map_name: Map name identifier  
            cluster: Cluster type ('brigit', 'cuenca', 'local')
            
        Returns:
            Dictionary containing all relevant paths
        """
        if cluster not in self.config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(self.config.cluster_paths.keys())}")
        
        local_path = self.config.cluster_paths[cluster]
        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/map_{map_name}"
        
        paths = {
            'raw_dir': raw_dir,
            'output_path': f"{raw_dir}/training_results.csv",
            'figures_dir': f"{raw_dir}/training_figures/",
            'smoothed_figures_dir': f"{raw_dir}/training_figures/smoothed_{self.config.smoothing_factor}/"
        }
        
        # Create directories
        base_dirs = [paths['raw_dir']]
        for dir_path in base_dirs + [paths['figures_dir'], paths['smoothed_figures_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        return paths
    
    def parse_training_folder(self, folder_path: str, num_agents: int = 1) -> Optional[pd.DataFrame]:
        """
        Parse a single training folder and extract data.
        
        Args:
            folder_path: Path to training folder
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
        # Check if 'episode' column exists and use it, otherwise use line numbers
        if 'episode' in df.columns:
            # Use the existing episode column as x-axis
            pass  # Keep the original episode values
        else:
            print(f"Warning: 'episode' column not found in {folder_path}. Using line numbers as episode values.")
            # If no episode column exists, create one with line numbers
            if df.shape[1] > 0:
                df.iloc[:, 0] = range(1, len(df) + 1)
                # Rename the first column to 'episode' if it's not already named
                if df.columns[0] != 'episode':
                    df.rename(columns={df.columns[0]: 'episode'}, inplace=True)
            else:
                # If dataframe is empty or has no columns, create an episode column
                df['episode'] = range(1, len(df) + 1)
        
        # Extract folder timestamp
        folder_name = os.path.basename(folder_path)
        date_time_str = folder_name.replace("Training_", "")
        
        # Add metadata columns
        df.insert(0, "timestamp", date_time_str)
        
        # Add agent parameters
        for i, (name, alpha, beta) in enumerate(matches, 1):
            df.insert(0 + i, f"alpha_{i}", float(alpha))
            df.insert(1 + i, f"beta_{i}", float(beta))
        
        df.insert(len(matches) * 2 + 1, "lr", lr)
        
        return df
    
    def load_experiment_data(self, paths: Dict[str, str], 
                           num_agents: int = 1) -> pd.DataFrame:
        """
        Load all experiment data from the raw directory.
        
        Args:
            paths: Dictionary containing directory paths
            num_agents: Number of agents in the experiment
            
        Returns:
            Combined DataFrame with all experiment data
        """
        all_dfs = []
        
        raw_dir = paths['raw_dir']
        print("Loading experiment data from directory:", raw_dir)

        if os.path.exists(raw_dir):                        
            for folder in os.listdir(raw_dir):
                folder_path = os.path.join(raw_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                
                df = self.parse_training_folder(folder_path, num_agents)
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
            df["total_deliveries"] = df["deliver_ai_rl_1"]
        else:  # num_agents == 2
            df["attitude_key"] = df.apply(
                lambda row: f"{row['alpha_1']}_{row['beta_1']}_{row['alpha_2']}_{row['beta_2']}", 
                axis=1
            )
            df["pure_reward_total"] = df["pure_reward_ai_rl_1"] + df["pure_reward_ai_rl_2"]
            df["total_deliveries"] = df["deliver_ai_rl_1"] + df["deliver_ai_rl_2"]
        
        return df


class MetricDefinitions:
    """Defines metrics and their visual properties for different experiment types."""
    
    @staticmethod
    def get_base_classic_metrics() -> Dict:
        """Get base metric definitions for classic experiments (without agent suffix)."""
        return {
            'rewarded_metrics': [
                "deliver",
                "cut",
                "salad",
                "plate",
                "raw_food",
                "counter"
            ],
            'movement_metrics': [
                "do_nothing",
                "useless_floor",
                "useless_wall",
                "useless_counter",
                "useful_counter",
                "salad_assembly",
                "destructive_food_dispenser",
                "useful_food_dispenser",
                "useless_cutting_board",
                "useful_cutting_board",
                "destructive_plate_dispenser",
                "useful_plate_dispenser",
                "useless_delivery",
                "useful_delivery",
                "inaccessible_tile"
            ]
        }
    
    @staticmethod
    def get_classic_metrics() -> Dict:
        """Get metric definitions for classic experiments (for backward compatibility)."""
        base_metrics = MetricDefinitions.get_base_classic_metrics()
        return {
            'rewarded_metrics_1': [f"{metric}_ai_rl_1" for metric in base_metrics['rewarded_metrics']],
            'movement_metrics_1': [f"{metric}_ai_rl_1" for metric in base_metrics['movement_metrics']]
        }
    
    @staticmethod
    def get_agent_metrics(base_metrics: List[str], agent_num: int) -> List[str]:
        """Generate agent-specific metrics from base metric names."""
        return [f"{metric}_ai_rl_{agent_num}" for metric in base_metrics]
    
    @staticmethod
    def get_base_competition_metrics() -> Dict:
        """Get base metric definitions for competition experiments (without agent suffix)."""
        return {
            'result_events': [
                "deliver_own",
                "deliver_other", 
                "salad_own",
                "salad_other",
                "cut_own",
                "cut_other",
            ],
            'action_types': [
                "do_nothing",
                "floor_actions",
                "wall_actions",
                "useless_counter_actions",
                "useful_counter_actions",
                "useless_own_food_dispenser_actions",
                "useful_own_food_dispenser_actions",
                "useless_other_food_dispenser_actions",
                "useful_other_food_dispenser_actions",
                "useless_cutting_board_actions",
                "useful_own_cutting_board_actions",
                "useful_other_cutting_board_actions",
                "useless_plate_dispenser_actions",
                "useful_plate_dispenser_actions",
                "useless_delivery_actions",
                "useful_own_delivery_actions",
                "useful_other_delivery_actions",
            ]
        }
    
    @staticmethod
    def get_competition_metrics() -> Dict:
        """Get metric definitions for competition experiments (for backward compatibility)."""
        base_metrics = MetricDefinitions.get_base_competition_metrics()
        return {
            'result_events_1': [f"{metric}_ai_rl_1" for metric in base_metrics['result_events']],
            'result_events_2': [f"{metric}_ai_rl_2" for metric in base_metrics['result_events']],
            'action_types_1': [f"{metric}_ai_rl_1" for metric in base_metrics['action_types']],
            'action_types_2': [f"{metric}_ai_rl_2" for metric in base_metrics['action_types']]
        }
    
    @staticmethod
    def get_metric_labels() -> Dict:
        """Get human-readable labels for metrics."""
        return {
            # Classic metrics
            "deliver": "Delivered",
            "cut": "Cut", 
            "salad": "Salad",
            "plate": "Plate",
            "raw_food": "Raw Food",
            "counter": "Counter",
            "do_nothing": "No action",
            "useless_floor": "Useless Floor",
            "useless_wall": "Useless Wall",
            "useless_counter": "Useless Counter",
            "useful_counter": "Useful Counter",
            "salad_assembly": "Salad Assembly",
            "destructive_food_dispenser": "Destructive Food Dispenser",
            "useful_food_dispenser": "Useful Food Dispenser",
            "useless_cutting_board": "Useless Cutting Board",
            "useful_cutting_board": "Useful Cutting Board",
            "destructive_plate_dispenser": "Destructive Plate Dispenser",
            "useful_plate_dispenser": "Useful Plate Dispenser",
            "useless_delivery": "Useless Delivery",
            "useful_delivery": "Useful Delivery",
            "inaccessible_tile": "Inaccessible Tile",
            
            # Legacy classic metrics for compatibility
            "delivered": "Delivered",
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
            "deliver_own": "Delivered Own",
            "deliver_other": "Delivered Other",
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
            "deliver": "#27AE60",
            "cut": "#2980B9",
            "salad": "#E67E22",
            "plate": "#F39C12",
            "raw_food": "#8E44AD",
            "counter": "#34495E",
            "do_nothing": "#000000",
            "useless_floor": "#9B59B6",
            "useless_wall": "#59351F",
            "useless_counter": "#D5D8DC",
            "useful_counter": "#7B7D7D",
            "salad_assembly": "#A569BD",
            "destructive_food_dispenser": "#E74C3C",
            "useful_food_dispenser": "#C0392B",
            "useless_cutting_board": "#AED6F1",
            "useful_cutting_board": "#2980B9",
            "destructive_plate_dispenser": "#FF6B35",
            "useful_plate_dispenser": "#E67E22",
            "useless_delivery": "#A9DFBF",
            "useful_delivery": "#27AE60",
            "inaccessible_tile": "#95A5A6",
            
            # Legacy classic colors for compatibility
            "delivered": "#27AE60",
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
            "deliver_own": "#A9DFBF",
            "deliver_other": "#27AE60",
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
        
        if by_attitude:
            for attitude in unique_attitudes:
                subset = df[df["attitude_key"] == attitude]
                
                plt.figure(figsize=(10, 6))
                
                for lr in unique_lr:
                    lr_filtered = subset[subset["lr"] == lr]
                    grouped = lr_filtered.groupby("episode")[metric_col].mean().reset_index()
                    label = f"LR {lr}"
                    plt.plot(grouped["episode"], grouped[metric_col], label=label)
                
                plt.title(f"{metric_name} vs Episode\nAttitude {attitude}")
                plt.xlabel("Episode")
                plt.ylabel(metric_name)
                plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = self._sanitize_filename(attitude)
                filename = f"{metric_name.lower().replace(' ', '_')}_attitude_{sanitized_attitude}.png"
                plt.savefig(os.path.join(figures_dir, filename))
                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            
            for lr in unique_lr:
                lr_filtered = df[df["lr"] == lr]
                grouped = lr_filtered.groupby("episode")[metric_col].mean().reset_index()
                
                label = f"LR {lr}"
                plt.plot(grouped["episode"], grouped[metric_col], label=label)
            
            plt.xlabel("Episodes", fontsize=20)
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
        
        if by_attitude:
            for attitude in unique_attitudes:
                subset = df[df["attitude_key"] == attitude]
                
                plt.figure(figsize=(10, 6))
                
                for lr in unique_lr:
                    lr_filtered = subset[subset["lr"] == lr]
                    lr_filtered = lr_filtered.copy()
                    lr_filtered["episode_block"] = (lr_filtered["episode"] // N)
                    block_means = lr_filtered.groupby("episode_block")[metric_col].mean()
                    middle_episodes = lr_filtered.groupby("episode_block")["episode"].median()
                    
                    label = f"LR {lr}"
                    plt.plot(middle_episodes, block_means, label=label)
                
                plt.title(f"{metric_name} vs Episode\nAttitude {attitude}")
                plt.xlabel("Episode")
                plt.ylabel(metric_name)
                plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = self._sanitize_filename(attitude)
                filename = f"{metric_name.lower().replace(' ', '_')}_attitude_{sanitized_attitude}_smoothed_{N}.png"
                plt.savefig(os.path.join(figures_dir, filename))
                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            
            for lr in unique_lr:
                lr_filtered = df[df["lr"] == lr]
                lr_filtered = lr_filtered.copy()
                lr_filtered["episode_block"] = (lr_filtered["episode"] // N)
                block_means = lr_filtered.groupby("episode_block")[metric_col].mean()
                middle_episodes = lr_filtered.groupby("episode_block")["episode"].median()
                
                label = f"LR {lr}"
                plt.plot(middle_episodes, block_means, label=label)
            
            plt.xlabel("Episodes", fontsize=20)
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
        unique_lr = df["lr"].unique()
        
        for lr in unique_lr:
            filtered_subset = df[df["lr"] == lr]
            
            if smoothed:
                filtered_subset = filtered_subset.copy()
                filtered_subset["episode_block"] = (filtered_subset["episode"] // N)
            
            plt.figure(figsize=(12, 6))
            
            for metric in metrics:
                label, color = self._get_metric_info(metric)
                
                if smoothed:
                    block_means = filtered_subset.groupby("episode_block")[metric].mean()
                    middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                    plt.plot(middle_episodes, block_means, label=label, color=color)
                else:
                    grouped = filtered_subset.groupby("episode")[metric].mean().reset_index()
                    plt.plot(grouped["episode"], grouped[metric], label=label, color=color)
            
            title = f"Metrics per Episode - LR {lr}"
            if smoothed:
                title += f" (Smoothed {N})"
            
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel("Mean value")
            plt.legend()
            plt.tight_layout()
            
            suffix = f"_smoothed_{N}" if smoothed else ""
            filename = f"metrics_agent{agent_num}_lr{str(lr).replace('.', 'p')}{suffix}.png"
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
    
    parser.add_argument(
        '--individual-trainings',
        type=str,
        choices=['yes', 'no', 'Yes', 'No', 'YES', 'NO'],
        default='no',
        help='Generate individual figures for each training ID (yes/no)'
    )
    
    return parser


def main_analysis_pipeline(experiment_type: str, map_name: str,
                          cluster: str = 'cuenca', smoothing_factor: int = 15, 
                          num_agents: Optional[int] = None) -> Dict:
    """
    Main analysis pipeline that can be used by all experiment types.
    
    Args:
        experiment_type: Type of experiment
        map_name: Map name
        cluster: Cluster type
        smoothing_factor: Smoothing factor for plots
        num_agents: Number of agents (default: 2 for classic/competition, 1 for pretrained)
        
    Returns:
        Dictionary containing processed data and paths
    """
    # Initialize components
    config = AnalysisConfig()
    config.smoothing_factor = smoothing_factor
    
    processor = DataProcessor(config)
    plotter = PlotGenerator(config)
    
    # Set up directories and load data
    paths = processor.setup_directories(experiment_type, map_name, cluster)
    
    # Determine number of agents based on experiment type
    if num_agents is None:
        # Pretrained experiments have 1 agent, others have 2
        num_agents = 1 if 'pretrain' in experiment_type.lower() else 2
    
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