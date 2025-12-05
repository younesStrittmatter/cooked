#!/usr/bin/env python3
"""
Grid Search Results Analysis Tool
=================================

This script analyzes the performance of different hyperparameter configurations
from DTDE grid search experiments. It provides comprehensive analysis including:
- Performance metrics extraction and comparison
- Statistical analysis of hyperparameter effects  
- Visualization of results
- Best configuration identification

Usage:
    python analyze_grid_search_results.py <results_directory>
    python analyze_grid_search_results.py --help

Author: Grid Search Analysis Tool
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GridSearchAnalyzer:
    """Main class for analyzing grid search results"""
    
    def __init__(self, results_dir: str, output_dir: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir
        self.results_df = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def parse_experiment_id(self, exp_id: str) -> Dict[str, str]:
        """Parse experiment ID to extract configuration parameters"""
        config = {'experiment_id': exp_id}
        
        # Split by underscore and parse each component
        parts = exp_id.split('_')
        
        for part in parts:
            if part.startswith('lr'):
                config['learning_rate'] = part.replace('lr', '')
            elif part.startswith('seed'):
                config['seed'] = part.replace('seed', '')
            elif part.startswith('batch'):
                config['batch_size'] = part.replace('batch', '')
            elif part in ['low', 'medium', 'high']:
                config['penalty_level'] = part
            elif part in ['sparse', 'shaped', 'dense', 'equal']:
                config['reward_type'] = part
            elif part.startswith('exp'):
                config['experiment_number'] = part.replace('exp', '')
        
        return config
    
    def parse_training_log(self, log_file: Path) -> Dict[str, float]:
        """Extract training metrics from log file"""
        metrics = {}
        
        if not log_file.exists():
            logger.warning(f"Log file not found: {log_file}")
            return metrics
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract various metrics based on common patterns
            patterns = {
                'final_reward': [
                    r'final.*?reward.*?([+-]?\d+\.?\d*)',
                    r'episode.*?reward.*?([+-]?\d+\.?\d*)',
                    r'total.*?reward.*?([+-]?\d+\.?\d*)'
                ],
                'max_reward': [
                    r'max.*?reward.*?([+-]?\d+\.?\d*)',
                    r'best.*?reward.*?([+-]?\d+\.?\d*)'
                ],
                'convergence_episode': [
                    r'converged.*?episode.*?(\d+)',
                    r'convergence.*?(\d+).*?episode'
                ],
                'training_time': [
                    r'training.*?time.*?(\d+\.?\d*)',
                    r'elapsed.*?time.*?(\d+\.?\d*)',
                    r'duration.*?(\d+\.?\d*)'
                ],
                'final_loss': [
                    r'final.*?loss.*?([+-]?\d+\.?\d*)',
                    r'loss.*?([+-]?\d+\.?\d*).*?final'
                ]
            }
            
            for metric_name, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            metrics[metric_name] = value
                            break  # Use first successful match
                        except (ValueError, IndexError):
                            continue
            
            # Extract episode-wise data for more detailed analysis
            episode_rewards = re.findall(r'episode.*?(\d+).*?reward.*?([+-]?\d+\.?\d*)', content, re.IGNORECASE)
            if episode_rewards:
                rewards = [float(r[1]) for r in episode_rewards[-10:]]  # Last 10 episodes
                if rewards:
                    metrics['avg_final_episodes'] = np.mean(rewards)
                    metrics['std_final_episodes'] = np.std(rewards)
            
        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {e}")
        
        return metrics
    
    def collect_all_results(self) -> pd.DataFrame:
        """Collect all experiment results into a DataFrame"""
        results = []
        
        # Find all output files
        output_files = list(self.results_dir.glob('output_exp*.txt'))
        
        if not output_files:
            logger.warning("No experiment output files found!")
            return pd.DataFrame()
        
        logger.info(f"Found {len(output_files)} experiment output files")
        
        for output_file in output_files:
            # Extract experiment ID from filename
            exp_id = output_file.stem.replace('output_', '')
            
            # Parse configuration from experiment ID
            config = self.parse_experiment_id(exp_id)
            
            # Parse training metrics from log
            metrics = self.parse_training_log(output_file)
            
            # Combine configuration and metrics
            result = {**config, **metrics}
            
            # Add file metadata
            result['log_file'] = str(output_file)
            result['file_size'] = output_file.stat().st_size
            result['last_modified'] = output_file.stat().st_mtime
            
            results.append(result)
        
        df = pd.DataFrame(results)
        logger.info(f"Collected {len(df)} experiment results")
        
        return df
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the grid search"""
        if self.results_df is None or self.results_df.empty:
            return {}
        
        stats = {}
        
        # Basic experiment counts
        stats['total_experiments'] = len(self.results_df)
        stats['successful_experiments'] = self.results_df['final_reward'].notna().sum()
        stats['success_rate'] = stats['successful_experiments'] / stats['total_experiments']
        
        # Performance metrics
        if 'final_reward' in self.results_df.columns:
            reward_stats = self.results_df['final_reward'].describe()
            stats['reward_stats'] = reward_stats.to_dict()
        
        # Configuration analysis
        categorical_cols = ['penalty_level', 'reward_type', 'learning_rate']
        for col in categorical_cols:
            if col in self.results_df.columns:
                stats[f'{col}_distribution'] = self.results_df[col].value_counts().to_dict()
        
        return stats
    
    def find_best_configurations(self, top_k: int = 10) -> pd.DataFrame:
        """Find the top-k best performing configurations"""
        if self.results_df is None or 'final_reward' not in self.results_df.columns:
            return pd.DataFrame()
        
        # Sort by final reward (assuming higher is better)
        best_configs = self.results_df.nlargest(top_k, 'final_reward')
        
        # Select relevant columns for display
        display_cols = ['experiment_id', 'final_reward', 'learning_rate', 
                       'penalty_level', 'reward_type', 'convergence_episode', 'training_time']
        
        # Only include columns that exist
        available_cols = [col for col in display_cols if col in best_configs.columns]
        
        return best_configs[available_cols]
    
    def create_visualizations(self):
        """Create comprehensive visualizations of results"""
        if self.results_df is None or self.results_df.empty:
            logger.warning("No data available for visualization")
            return
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Reward distribution histogram
        if 'final_reward' in self.results_df.columns:
            plt.figure(figsize=(10, 6))
            self.results_df['final_reward'].hist(bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Final Rewards')
            plt.xlabel('Final Reward')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / 'reward_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Performance by hyperparameter
        categorical_params = ['penalty_level', 'reward_type', 'learning_rate']
        
        for param in categorical_params:
            if param in self.results_df.columns and 'final_reward' in self.results_df.columns:
                plt.figure(figsize=(10, 6))
                
                # Box plot
                self.results_df.boxplot(column='final_reward', by=param, ax=plt.gca())
                plt.title(f'Final Reward by {param.replace("_", " ").title()}')
                plt.suptitle('')  # Remove automatic title
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'reward_by_{param}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Correlation heatmap (for numerical columns)
        numerical_cols = self.results_df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.results_df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Correlation Matrix of Numerical Parameters')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Training convergence (if convergence data available)
        if 'convergence_episode' in self.results_df.columns:
            plt.figure(figsize=(10, 6))
            self.results_df['convergence_episode'].hist(bins=20, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Convergence Episodes')
            plt.xlabel('Episodes to Convergence')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / 'convergence_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report"""
        if self.results_df is None:
            return "No data available for analysis"
        
        report = []
        report.append("=" * 60)
        report.append("GRID SEARCH RESULTS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        stats = self.generate_summary_statistics()
        
        report.append("EXPERIMENT SUMMARY:")
        report.append("-" * 30)
        report.append(f"Total Experiments: {stats.get('total_experiments', 'N/A')}")
        report.append(f"Successful Experiments: {stats.get('successful_experiments', 'N/A')}")
        report.append(f"Success Rate: {stats.get('success_rate', 0):.2%}")
        report.append("")
        
        # Performance statistics
        if 'reward_stats' in stats:
            report.append("PERFORMANCE STATISTICS:")
            report.append("-" * 30)
            reward_stats = stats['reward_stats']
            report.append(f"Mean Reward: {reward_stats.get('mean', 0):.4f}")
            report.append(f"Std Deviation: {reward_stats.get('std', 0):.4f}")
            report.append(f"Best Reward: {reward_stats.get('max', 0):.4f}")
            report.append(f"Worst Reward: {reward_stats.get('min', 0):.4f}")
            report.append("")
        
        # Best configurations
        best_configs = self.find_best_configurations(5)
        if not best_configs.empty:
            report.append("TOP 5 CONFIGURATIONS:")
            report.append("-" * 30)
            for idx, (_, row) in enumerate(best_configs.iterrows(), 1):
                report.append(f"{idx}. Experiment: {row.get('experiment_id', 'N/A')}")
                report.append(f"   Final Reward: {row.get('final_reward', 'N/A'):.4f}")
                report.append(f"   Learning Rate: {row.get('learning_rate', 'N/A')}")
                report.append(f"   Penalty Level: {row.get('penalty_level', 'N/A')}")
                report.append(f"   Reward Type: {row.get('reward_type', 'N/A')}")
                report.append("")
        
        # Configuration analysis
        for param in ['penalty_level', 'reward_type', 'learning_rate']:
            if f'{param}_distribution' in stats:
                report.append(f"{param.replace('_', ' ').upper()} DISTRIBUTION:")
                report.append("-" * 30)
                for value, count in stats[f'{param}_distribution'].items():
                    report.append(f"{value}: {count} experiments")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self):
        """Save all analysis results to files"""
        # Save DataFrame to CSV
        if self.results_df is not None:
            csv_path = self.output_dir / 'grid_search_results.csv'
            self.results_df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to: {csv_path}")
        
        # Save text report
        report = self.generate_report()
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Analysis report saved to: {report_path}")
        
        # Create visualizations
        self.create_visualizations()
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("Starting grid search analysis...")
        
        # Collect all results
        self.results_df = self.collect_all_results()
        
        if self.results_df.empty:
            logger.error("No results found to analyze!")
            return False
        
        # Generate and save all analysis outputs
        self.save_results()
        
        # Print summary to console
        print(self.generate_report())
        
        logger.info("Analysis completed successfully!")
        return True


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Analyze grid search results from DTDE training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_grid_search_results.py ./grid_search_logs/experiment_20231202
    python analyze_grid_search_results.py /path/to/results --output /path/to/analysis
        """
    )
    
    parser.add_argument(
        'results_dir',
        help='Directory containing grid search results'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for analysis results (default: same as results_dir)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Run analysis
    analyzer = GridSearchAnalyzer(args.results_dir, args.output)
    success = analyzer.run_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()