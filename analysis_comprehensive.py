#!/usr/bin/env python3
"""
Comprehensive Analysis Script for spoiled_broth experiments.

This script orchestrates the complete analysis by calling:
1. analysis_simulations.py - for detailed individual simulation analysis
2. analysis_checkpoint_comparison.py - for checkpoint comparison across training_ids

Both analyses save their outputs to the same organized directory: map_{map_nr}/simulation_figures/

Usage:
nohup python3 analysis_comprehensive.py --cluster cuenca --map_nr baseline_division_of_labor --game_version classic --intent_version v3.1 --cooperative 1 > log_analysis_comprehensive.out 2>&1 &

Optional arguments:
--training_id and --checkpoint_number: If provided, run detailed simulation analysis for that specific combination
--output_dir: Custom output directory (default: map_{map_nr}/simulation_figures/)

Author: Samuel Lozano
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import time


class ComprehensiveAnalysisOrchestrator:
    """Main class that orchestrates both simulation and checkpoint analyses."""
    
    def __init__(self, base_cluster_dir="", map_nr=None, game_version="classic", 
                 intent_version="v3.1", cooperative=1, training_id=None, 
                 checkpoint_number=None, output_dir=None):
        """
        Initialize the comprehensive analysis orchestrator.
        
        Args:
            base_cluster_dir: Base cluster directory 
            map_nr: Map number/name (e.g., "baseline_division_of_labor")
            game_version: Game version ("classic" or "competition")
            intent_version: Intent version (e.g., "v3.1")
            cooperative: 1 for cooperative, 0 for competitive
            training_id: Optional specific training ID for detailed analysis
            checkpoint_number: Optional specific checkpoint for detailed analysis
            output_dir: Custom output directory
        """
        self.base_cluster_dir = base_cluster_dir
        self.map_nr = map_nr
        self.game_version = game_version
        self.intent_version = intent_version
        self.cooperative = cooperative
        self.training_id = training_id
        self.checkpoint_number = checkpoint_number
        self.output_dir = output_dir
        
        # Determine the output directory
        if output_dir is None:
            map_base_dir = Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{map_nr}/")
            self.output_dir = map_base_dir / "simulation_figures"
        else:
            self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get paths to the analysis scripts
        self.script_dir = Path(__file__).parent
        self.simulations_script = self.script_dir / "analysis_simulations.py"
        self.checkpoint_script = self.script_dir / "analysis_checkpoint_comparison.py"
    
    def find_all_training_directories(self):
        """Find all training directories for the specified map."""
        cooperative_dir = 'cooperative' if self.cooperative == 1 else 'competitive'
        base_map_dir = Path(f"{self.base_cluster_dir}/data/samuel_lozano/cooked/{self.game_version}/{self.intent_version}/map_{self.map_nr}/{cooperative_dir}")
        
        training_dirs = []
        
        if not base_map_dir.exists():
            print(f"Error: Map directory does not exist: {base_map_dir}")
            return training_dirs
        
        print(f"Searching for training directories in: {base_map_dir}")
        
        # Search for Training_* directories
        for training_dir in base_map_dir.glob("Training_*"):
            if training_dir.is_dir():
                training_id = training_dir.name.replace("Training_", "")
                training_dirs.append(training_id)
                print(f"  Found training: {training_id}")
        
        print(f"Found {len(training_dirs)} training directories")
        return training_dirs
    
    def run_single_simulation_analysis(self, training_id, checkpoint_number="final"):
        """Run simulation analysis for a single training_id and checkpoint."""
        print(f"Running analysis_simulations.py for Training {training_id}, Checkpoint {checkpoint_number}")
        
        # Build command for analysis_simulations.py
        cmd = [
            "python3", str(self.simulations_script),
            "--map_nr", self.map_nr,
            "--training_id", training_id,
            "--checkpoint_number", checkpoint_number,
            "--game_version", self.game_version,
            "--intent_version", self.intent_version,
            "--cooperative", str(self.cooperative)
        ]
        
        # Determine cluster argument
        if self.base_cluster_dir == "":
            cmd.extend(["--cluster", "cuenca"])
        elif "brigit" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "brigit"])
        elif "onedrive" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "local"])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the simulations analysis
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print(f"✅ Simulation analysis for Training {training_id} completed successfully!")
                print("Output:")
                print(result.stdout)
                return True
            else:
                print(f"❌ Simulation analysis for Training {training_id} failed!")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Simulation analysis for Training {training_id} timed out (>1 hour)")
            return False
        except Exception as e:
            print(f"❌ Error running simulation analysis for Training {training_id}: {e}")
            return False
    
    def run_simulations_analysis(self):
        """Run the detailed simulations analysis."""
        print("PHASE 1: DETAILED SIMULATION ANALYSIS")
        print("-" * 40)
        
        if self.training_id:
            # Specific training_id provided - run only for that one
            checkpoint = self.checkpoint_number if self.checkpoint_number else "final"
            print(f"Running simulation analysis for specific Training {self.training_id}, Checkpoint {checkpoint}")
            return self.run_single_simulation_analysis(self.training_id, checkpoint)
        else:
            # No specific training_id - run for all training_ids found
            training_ids = self.find_all_training_directories()
            
            if not training_ids:
                print("❌ No training directories found!")
                return False
            
            print(f"Running simulation analysis for all {len(training_ids)} training directories with checkpoint 'final'")
            
            success_count = 0
            for training_id in training_ids:
                print(f"\n--- Processing Training {training_id} ---")
                if self.run_single_simulation_analysis(training_id, "final"):
                    success_count += 1
                else:
                    print(f"⚠️  Training {training_id} failed, continuing with next...")
            
            print(f"\n📊 Simulation Analysis Summary: {success_count}/{len(training_ids)} training directories completed successfully")
            
            # Consider it successful if at least half completed successfully
            return success_count >= len(training_ids) // 2
    
    def run_checkpoint_analysis(self):
        """Run the checkpoint comparison analysis."""
        print("\nPHASE 2: CHECKPOINT COMPARISON ANALYSIS")
        print("-" * 40)
        print(f"Running analysis_checkpoint_comparison.py for map {self.map_nr}")
        
        # Build command for analysis_checkpoint_comparison.py
        cmd = [
            "python3", str(self.checkpoint_script),
            "--map_nr", self.map_nr,
            "--game_version", self.game_version,
            "--intent_version", self.intent_version,
            "--cooperative", str(self.cooperative),
            "--output_dir", str(self.output_dir)
        ]
        
        # Determine cluster argument
        if self.base_cluster_dir == "":
            cmd.extend(["--cluster", "cuenca"])
        elif "brigit" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "brigit"])
        elif "onedrive" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "local"])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the checkpoint analysis
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print("✅ Checkpoint comparison analysis completed successfully!")
                print("Output:")
                print(result.stdout)
                return True
            else:
                print("❌ Checkpoint comparison analysis failed!")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Checkpoint comparison analysis timed out (>1 hour)")
            return False
        except Exception as e:
            print(f"❌ Error running checkpoint comparison analysis: {e}")
            return False
    
    def generate_comprehensive_summary(self, simulations_success, checkpoint_success):
        """Generate a comprehensive summary combining both analyses."""
        print("\nPHASE 3: COMPREHENSIVE SUMMARY")
        print("-" * 40)
        
        summary = []
        summary.append("# Comprehensive Analysis Summary Report")
        summary.append("## Spoiled Broth Experiments\n")
        summary.append(f"Generated on: {pd.Timestamp.now()}\n")
        
        # Analysis parameters
        summary.append("## Analysis Configuration")
        summary.append(f"- **Map**: {self.map_nr}")
        summary.append(f"- **Game version**: {self.game_version}")
        summary.append(f"- **Intent version**: {self.intent_version}")
        summary.append(f"- **Mode**: {'cooperative' if self.cooperative == 1 else 'competitive'}")
        summary.append(f"- **Base directory**: {self.base_cluster_dir}")
        if self.training_id and self.checkpoint_number:
            summary.append(f"- **Detailed analysis**: Training {self.training_id}, Checkpoint {self.checkpoint_number}")
        summary.append(f"- **Output directory**: {self.output_dir}")
        summary.append("")
        
        # Analysis phases completed
        summary.append("## Analysis Phases Completed")
        
        if simulations_success:
            summary.append("### ✅ Phase 1: Detailed Simulation Analysis")
            if self.training_id:
                checkpoint = self.checkpoint_number if self.checkpoint_number else "final"
                summary.append(f"- Analyzed individual simulations for Training {self.training_id}, Checkpoint {checkpoint}")
            else:
                summary.append("- Analyzed individual simulations for all training directories")
                summary.append("- Processed all available training_ids with checkpoint 'final'")
            summary.append("- Generated detailed behavioral plots (deliveries, actions, distances)")
            summary.append("- Created individual and aggregated visualizations")
            summary.append("- Results saved in subdirectories of simulation_figures/")
        else:
            summary.append("### ❌ Phase 1: Detailed Simulation Analysis")
            summary.append("- Failed to complete (check logs for details)")
        
        if checkpoint_success:
            summary.append("\n### ✅ Phase 2: Checkpoint Comparison Analysis")
            summary.append("- Analyzed delivery performance across all training IDs and checkpoints")
            summary.append("- Generated comparative plots showing training progression")
            summary.append("- Created statistical summaries of checkpoint performance")
            summary.append("- Results saved directly in simulation_figures/")
        else:
            summary.append("\n### ❌ Phase 2: Checkpoint Comparison Analysis")
            summary.append("- Failed to complete (check logs for details)")
        
        summary.append("")
        
        # Generated files overview
        summary.append("## Generated Files Structure")
        summary.append(f"All outputs are organized in: `{self.output_dir}`")
        summary.append("")
        
        if checkpoint_success:
            summary.append("### Checkpoint Comparison Analysis")
            summary.append("- `deliveries_vs_checkpoints_{}.png` - Multi-panel checkpoint comparison".format(self.map_nr))
            summary.append("- `deliveries_vs_checkpoints_focused_{}.png` - Focused line plot".format(self.map_nr))
            summary.append("- `checkpoint_analysis_summary.md` - Detailed checkpoint analysis report")
        
        if simulations_success:
            summary.append("\n### Detailed Simulation Analysis")
            summary.append(f"- `detailed_simulations_{self.checkpoint_number}/` - Subdirectory containing:")
            summary.append("  - Individual simulation plots (deliveries, actions, distances)")
            summary.append("  - Aggregated simulation visualizations")
            summary.append("  - `detailed_simulation_analysis_summary.md` - Detailed simulation report")
        
        summary.append("\n### This Report")
        summary.append("- `comprehensive_analysis_summary.md` - This comprehensive overview")
        
        # Usage recommendations
        summary.append("\n## Usage Recommendations")
        summary.append("### For Research Analysis:")
        
        if checkpoint_success:
            summary.append("1. **Start with checkpoint comparison plots** to identify best-performing training setups")
            summary.append("2. **Use focused line plots** to understand training progression trends")
            summary.append("3. **Examine heatmaps** to compare training IDs across different checkpoints")
        
        if simulations_success:
            summary.append("4. **Dive into detailed simulation analysis** for behavioral insights")
            summary.append("5. **Compare individual vs aggregated plots** to understand variability")
        
        summary.append("\n### For Future Experiments:")
        summary.append("- **Monitor early checkpoint performance** for early stopping decisions")
        summary.append("- **Use this analysis framework** for systematic experiment evaluation")
        summary.append("- **Run comprehensive analysis regularly** to track training progress")
        
        # Analysis execution details
        summary.append("\n## Execution Details")
        summary.append("This comprehensive analysis was executed by calling:")
        if simulations_success:
            summary.append("1. `analysis_simulations.py` - for detailed simulation analysis")
        if checkpoint_success:
            summary.append("2. `analysis_checkpoint_comparison.py` - for checkpoint comparison")
        summary.append("\nBoth scripts were executed as subprocesses with organized output management.")
        
        # Save comprehensive summary
        summary_file = self.output_dir / 'comprehensive_analysis_summary.md'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary))
        
        print(f"Comprehensive summary report generated: {summary_file}")
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis by orchestrating both sub-analyses."""
        print("="*60)
        print("COMPREHENSIVE ANALYSIS - SPOILED BROTH EXPERIMENTS")
        print("="*60)
        print(f"Map: {self.map_nr}")
        print(f"Game version: {self.game_version}")
        print(f"Intent version: {self.intent_version}")
        print(f"Mode: {'cooperative' if self.cooperative == 1 else 'competitive'}")
        if self.training_id and self.checkpoint_number:
            print(f"Detailed analysis for: Training {self.training_id}, Checkpoint {self.checkpoint_number}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        print()
        
        # Track start time
        start_time = time.time()
        
        if self.training_id:
            # Specific training_id provided - only run simulation analysis
            print("🎯 Specific training_id provided - running detailed simulation analysis only")
            simulations_success = self.run_simulations_analysis()
            checkpoint_success = False  # Skip checkpoint analysis
        else:
            # No specific training_id - run simulations for all, then checkpoint comparison
            print("🔄 No specific training_id provided - running simulation analysis for all training_ids, then checkpoint comparison")
            
            # Phase 1: Detailed Simulation Analysis for all training_ids
            simulations_success = self.run_simulations_analysis()
            
            # Phase 2: Checkpoint Comparison Analysis
            checkpoint_success = self.run_checkpoint_analysis()
        
        # Phase 3: Generate comprehensive summary
        self.generate_comprehensive_summary(simulations_success, checkpoint_success)
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*60)
        print(f"Total execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        print(f"All results saved to: {self.output_dir}")
        print(f"Open {self.output_dir}/comprehensive_analysis_summary.md for complete results overview.")
        print()
        
        # Summary of what was completed
        if self.training_id:
            # Specific training_id mode
            if simulations_success:
                print(f"✅ Detailed simulation analysis for Training {self.training_id} completed successfully!")
            else:
                print(f"❌ Detailed simulation analysis for Training {self.training_id} failed. Check error messages above.")
        else:
            # All training_ids mode
            if simulations_success and checkpoint_success:
                print("✅ All analyses completed successfully!")
            elif checkpoint_success:
                print("⚠️  Checkpoint analysis completed, but some simulation analyses failed.")
            elif simulations_success:
                print("⚠️  Simulation analyses completed, but checkpoint analysis failed.")
            else:
                print("❌ Both simulation and checkpoint analyses failed. Check error messages above.")
        
        return simulations_success, checkpoint_success


def main():
    """Main function to run the comprehensive analysis."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive analysis of spoiled_broth experiments by orchestrating individual analysis scripts'
    )
    
    # Required arguments
    parser.add_argument('--map_nr', type=str, required=True,
                       help='Map number/name (e.g., "baseline_division_of_labor")')
    
    # Optional arguments with defaults
    parser.add_argument('--cluster', type=str, default='cuenca',
                       help='Base cluster (default: cuenca)')
    parser.add_argument('--game_version', type=str, default='classic', 
                       choices=['classic', 'competition'],
                       help='Game version (default: classic)')
    parser.add_argument('--intent_version', type=str, default='v3.1',
                       help='Intent version (default: v3.1)')
    parser.add_argument('--cooperative', type=int, default=1, choices=[0, 1],
                       help='1 for cooperative, 0 for competitive (default: 1)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: map_{map_nr}/{cooperative}/simulation_figures/)')
    
    # Optional arguments for detailed simulation analysis
    parser.add_argument('--training_id', type=str, default=None,
                       help='Optional: specific training ID for detailed simulation analysis')
    parser.add_argument('--checkpoint_number', type=str, default=None,
                       help='Optional: specific checkpoint for detailed simulation analysis (e.g., "final", "50")')

    args = parser.parse_args()

    # Set base cluster directory
    if args.cluster.lower() == 'cuenca':
        base_cluster_dir = ""
    elif args.cluster.lower() == 'brigit':
        base_cluster_dir = "/mnt/lustre/home/samuloza/"
    elif args.cluster.lower() == 'local':
        base_cluster_dir = "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
    else:
        print(f"Unknown cluster: {args.cluster}")
        return 1

    # Create orchestrator and run comprehensive analysis
    orchestrator = ComprehensiveAnalysisOrchestrator(
        base_cluster_dir=base_cluster_dir,
        map_nr=args.map_nr,
        game_version=args.game_version,
        intent_version=args.intent_version,
        cooperative=args.cooperative,
        training_id=args.training_id,
        checkpoint_number=args.checkpoint_number,
        output_dir=args.output_dir
    )

    simulations_success, checkpoint_success = orchestrator.run_comprehensive_analysis()
    
    # Return appropriate exit code
    if args.training_id:
        # Specific training_id mode - only simulation analysis matters
        return 0 if simulations_success else 1
    else:
        # All training_ids mode - both analyses matter
        if simulations_success and checkpoint_success:
            return 0  # All good
        elif checkpoint_success:
            return 0  # Checkpoint analysis completed, some simulations may have failed but that's acceptable
        else:
            return 1  # Major failure


if __name__ == "__main__":
    sys.exit(main())
