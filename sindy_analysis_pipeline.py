#!/usr/bin/env python3
"""
SINDy Analysis Pipeline for Spoiled Broth Simulations - Simplified Position Dynamics

This script provides a simplified pipeline to:
1. Extract CSV data (simulation.csv) based on parameters
2. Preprocess the data for SINDy analysis (using only x, y coordinates)
3. Apply Sparse Identification of Nonlinear Dynamics (SINDy) to discover governing equations for position dynamics
4. Save all results, plots, and analysis summaries in organized folders

The simplified version focuses only on x, y position coordinates and treats frame numbers as time.
This makes the analysis more focused on discovering the underlying spatial dynamics without
the complexity of discrete action inputs.

The script automatically creates a 'sindy_models/' directory within the simulation path
and saves the following for each agent:
- equations.txt: Discovered equations for x, y dynamics in human-readable format
- coefficients.csv: Model coefficients matrix
- metadata.json: Analysis metadata and parameters
- trajectory_comparison.png: Plots comparing true vs predicted x, y trajectories

Additionally, it creates overall analysis summaries:
- analysis_summary.json/txt: Overall analysis metadata and scores
- overview_comparison.png: Comparison plots across all agents
- summary_table.png: Summary table with key metrics

Usage:
# Single simulation:
nohup python sindy_analysis_pipeline.py --map_nr baseline_division_of_labor --training_id 2025-09-13_13-27-52 --checkpoint_number final --simulation_id 2025_10_02-09_38_20 > log_sindy.out 2>&1 &

# Combined simulations (recommended for better analysis):
nohup python sindy_analysis_pipeline.py --map_nr baseline_division_of_labor --training_id 2025-09-13_13-27-52 --checkpoint_number final --combine_simulations > log_sindy.out 2>&1 &

Example with all parameters:
nohup python sindy_analysis_pipeline.py --map_nr baseline_division_of_labor --training_id 2025-09-13_13-27-52 --checkpoint_number final --combine_simulations --cluster cuenca --game_version classic --intent_version v3.1 --cooperative True --smooth_data True --sindy_threshold 0.1 --polynomial_degree 2 --save_plots True > log_sindy.out 2>&1 &

Author: Samuel Lozano
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Import SINDy libraries
try:
    import pysindy as ps
    SINDY_AVAILABLE = True
except ImportError:
    print("Warning: pysindy not available. Install with: pip install pysindy")
    SINDY_AVAILABLE = False

warnings.filterwarnings('ignore')

class CSVDataExtractor:
    """Extract and organize CSV data from simulation directories."""
    
    def __init__(self, cluster="cuenca"):
        """
        Initialize the data extractor.
        
        Args:
            cluster: Cluster name (cuenca, local, etc.)
        """
        self.cluster = cluster
        
    def get_simulation_path(self, map_nr: str, training_id: str, checkpoint_number: str, 
                          simulation_id: str, game_version: str = "classic", 
                          intent_version: str = "v3.1", cooperative: bool = True) -> Path:
        """
        Construct the path to a specific simulation.
        
        Args:
            map_nr: Map number/name
            training_id: Training ID
            checkpoint_number: Checkpoint number
            simulation_id: Simulation ID
            game_version: Game version (classic/competition)
            intent_version: Intent version
            cooperative: Whether cooperative or competitive
            
        Returns:
            Path to the simulation directory
        """
        cooperative_dir = "cooperative" if cooperative else "competitive"
        
        if self.cluster == "cuenca":
            base_cluster_dir = ""
        elif self.cluster == "brigit":
            base_cluster_dir = "/mnt/lustre/home/samuloza/"
        elif self.cluster == "local":
            base_cluster_dir = "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
        else:
            base_cluster_dir = ""
            
        simulation_path = Path(
            f"{base_cluster_dir}/data/samuel_lozano/cooked/{game_version}/"
            f"{intent_version}/map_{map_nr}/{cooperative_dir}/Training_{training_id}/"
            f"simulations_{checkpoint_number}/simulation_{simulation_id}"
        )
        
        return simulation_path
    
    def find_all_simulations(self, map_nr: str, training_id: str, checkpoint_number: str, 
                           **kwargs) -> List[str]:
        """
        Find all simulation directories for given parameters.
        
        Args:
            map_nr: Map number/name
            training_id: Training ID
            checkpoint_number: Checkpoint number
            **kwargs: Additional parameters for path construction
            
        Returns:
            List of simulation IDs found
        """
        import glob
        
        # Get base path without simulation_id
        cooperative_dir = "cooperative" if kwargs.get('cooperative', True) else "competitive"
        game_version = kwargs.get('game_version', 'classic')
        intent_version = kwargs.get('intent_version', 'v3.1')
        
        if self.cluster == "cuenca":
            base_cluster_dir = ""
        elif self.cluster == "brigit":
            base_cluster_dir = "/mnt/lustre/home/samuloza/"
        elif self.cluster == "local":
            base_cluster_dir = "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
        else:
            base_cluster_dir = ""
            
        simulations_base_path = Path(
            f"{base_cluster_dir}/data/samuel_lozano/cooked/{game_version}/"
            f"{intent_version}/map_{map_nr}/{cooperative_dir}/Training_{training_id}/"
            f"simulations_{checkpoint_number}"
        )
        
        simulation_ids = []
        if simulations_base_path.exists():
            # Find all simulation_* directories
            pattern = str(simulations_base_path / "simulation_*")
            for sim_path in glob.glob(pattern):
                sim_dir = Path(sim_path)
                if sim_dir.is_dir():
                    # Extract simulation ID from directory name
                    sim_id = sim_dir.name.replace('simulation_', '')
                    # Check if both required CSV files exist
                    if (sim_dir / "simulation.csv").exists() and (sim_dir / "meaningful_actions.csv").exists():
                        simulation_ids.append(sim_id)
        
        print(f"Found {len(simulation_ids)} simulations: {simulation_ids}")
        return simulation_ids
    
    def extract_csv_data(self, map_nr: str, training_id: str, checkpoint_number: str, 
                        simulation_id: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Extract simulation.csv and meaningful_actions.csv for a specific simulation.
        
        Args:
            map_nr: Map number/name
            training_id: Training ID
            checkpoint_number: Checkpoint number
            simulation_id: Simulation ID
            **kwargs: Additional parameters for path construction
            
        Returns:
            Dictionary containing 'simulation' and 'actions' DataFrames
        """
        sim_path = self.get_simulation_path(map_nr, training_id, checkpoint_number, 
                                          simulation_id, **kwargs)
        
        sim_csv = sim_path / "simulation.csv"
        actions_csv = sim_path / "meaningful_actions.csv"
        
        data = {}
        
        if sim_csv.exists():
            data['simulation'] = pd.read_csv(sim_csv)
            print(f"Loaded simulation.csv: {len(data['simulation'])} rows")
        else:
            raise FileNotFoundError(f"simulation.csv not found at {sim_csv}")
            
        if actions_csv.exists():
            data['actions'] = pd.read_csv(actions_csv)
            print(f"Loaded meaningful_actions.csv: {len(data['actions'])} rows")
        else:
            print(f"Warning: meaningful_actions.csv not found at {actions_csv}")
            data['actions'] = pd.DataFrame()
            
        data['metadata'] = {
            'map_nr': map_nr,
            'training_id': training_id,
            'checkpoint_number': checkpoint_number,
            'simulation_id': simulation_id,
            'path': sim_path,
            'sindy_output_dir': sim_path / 'sindy_models'
        }
        
        return data
    
    def extract_all_simulations_data(self, map_nr: str, training_id: str, checkpoint_number: str, 
                                   **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Extract and combine data from all simulations for given parameters.
        
        Args:
            map_nr: Map number/name
            training_id: Training ID
            checkpoint_number: Checkpoint number
            **kwargs: Additional parameters for path construction
            
        Returns:
            Dictionary containing combined 'simulation' and 'actions' DataFrames
        """
        # Find all simulation IDs
        simulation_ids = self.find_all_simulations(map_nr, training_id, checkpoint_number, **kwargs)
        
        if not simulation_ids:
            raise FileNotFoundError(f"No simulations found for {map_nr}/{training_id}/{checkpoint_number}")
        
        all_simulation_data = []
        all_actions_data = []
        
        for sim_id in simulation_ids:
            try:
                data = self.extract_csv_data(map_nr, training_id, checkpoint_number, sim_id, **kwargs)
                
                # Add simulation ID to data for tracking
                sim_df = data['simulation'].copy()
                sim_df['simulation_id'] = sim_id
                all_simulation_data.append(sim_df)
                
                if not data['actions'].empty:
                    actions_df = data['actions'].copy()
                    actions_df['simulation_id'] = sim_id
                    all_actions_data.append(actions_df)
                    
                print(f"  Loaded simulation {sim_id}: {len(sim_df)} simulation rows, {len(actions_df) if not data['actions'].empty else 0} action rows")
                
            except Exception as e:
                print(f"  Warning: Could not load simulation {sim_id}: {e}")
                continue
        
        if not all_simulation_data:
            raise RuntimeError("No simulation data could be loaded")
        
        # Combine all data
        combined_simulation = pd.concat(all_simulation_data, ignore_index=True)
        combined_actions = pd.concat(all_actions_data, ignore_index=True) if all_actions_data else pd.DataFrame()
        
        print(f"Combined data: {len(combined_simulation)} simulation rows, {len(combined_actions)} action rows")
        
        # Get metadata from first simulation
        first_sim_path = self.get_simulation_path(map_nr, training_id, checkpoint_number, simulation_ids[0], **kwargs)
        
        return {
            'simulation': combined_simulation,
            'actions': combined_actions,
            'metadata': {
                'map_nr': map_nr,
                'training_id': training_id,
                'checkpoint_number': checkpoint_number,
                'simulation_ids': simulation_ids,
                'num_simulations': len(simulation_ids),
                'path': first_sim_path.parent,  # Use simulations_* directory
                'sindy_output_dir': first_sim_path.parent / 'sindy_models_combined'
            }
        }


class SINDyPreprocessor:
    """Preprocess data for SINDy analysis."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def align_timeframes(self, simulation_df: pd.DataFrame, 
                        actions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align simulation and actions data on the same timeframe.
        
        Args:
            simulation_df: Simulation data with continuous states
            actions_df: Actions data with discrete events
            
        Returns:
            Tuple of aligned DataFrames
        """
        if actions_df.empty:
            # Create dummy actions if no actions data
            max_frame = simulation_df['frame'].max()
            actions_aligned = pd.DataFrame({
                'frame': simulation_df['frame'].unique(),
                'action_type': 'none',
                'agent_id': 0
            })
        else:
            # Get frame range from simulation data
            min_frame = simulation_df['frame'].min()
            max_frame = simulation_df['frame'].max()
            
            # Filter actions to simulation timeframe
            actions_filtered = actions_df[
                (actions_df['frame'] >= min_frame) & 
                (actions_df['frame'] <= max_frame)
            ].copy()
            
            # For simplified version, we don't need aligned actions
            actions_aligned = pd.DataFrame()
        
        return simulation_df, actions_aligned
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            df: DataFrame with categorical columns
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # One-hot encode categorical variables
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def create_state_vectors(self, simulation_df: pd.DataFrame, 
                           actions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create state vectors X(t) for SINDy using only x,y coordinates.
        Properly handles multiple simulations by treating them as separate trajectories.
        
        Args:
            simulation_df: Aligned simulation data (potentially from multiple simulations)
            actions_df: Aligned actions data (not used in simplified version)
            
        Returns:
            List of agent data dictionaries with position trajectories
        """
        agent_data = []
        
        for agent_id in simulation_df['agent_id'].unique():
            agent_sim = simulation_df[simulation_df['agent_id'] == agent_id].copy()
            
            # Separate trajectories by simulation
            all_X = []
            all_t = []
            trajectory_info = []
            
            if 'simulation_id' in agent_sim.columns:
                # Stack all simulations into one long continuous trajectory
                all_sim_data = []
                time_offset = 0
                
                print(f"  Stacking {len(agent_sim['simulation_id'].unique())} simulations into one trajectory...")
                
                for sim_idx, sim_id in enumerate(sorted(agent_sim['simulation_id'].unique())):
                    sim_agent_data = agent_sim[agent_sim['simulation_id'] == sim_id].copy()
                    sim_agent_data = sim_agent_data.sort_values('frame')
                    
                    if len(sim_agent_data) < 10:  # Skip very short trajectories
                        print(f"    Skipping simulation {sim_id}: too short ({len(sim_agent_data)} points)")
                        continue
                    
                    # Adjust time to be continuous across simulations
                    sim_agent_data['frame_continuous'] = sim_agent_data['frame'] - sim_agent_data['frame'].min() + time_offset
                    
                    # Update time offset for next simulation
                    time_offset = sim_agent_data['frame_continuous'].max() + 1
                    
                    all_sim_data.append(sim_agent_data)
                    print(f"    Added simulation {sim_id}: {len(sim_agent_data)} points, time range [{sim_agent_data['frame_continuous'].min():.0f}, {sim_agent_data['frame_continuous'].max():.0f}]")
                
                if not all_sim_data:
                    print(f"  No valid simulations found for agent {agent_id}")
                    continue
                
                # Concatenate all simulation data
                combined_agent_data = pd.concat(all_sim_data, ignore_index=True)
                combined_agent_data = combined_agent_data.sort_values('frame_continuous')
                
                print(f"  Combined trajectory length: {len(combined_agent_data)} points")
                
                # Use position and velocity as state variables for richer dynamics
                state_cols = ['x', 'y']
                
                # Add velocity computed from position differences
                combined_agent_data['vx'] = combined_agent_data['x'].diff().fillna(0)
                combined_agent_data['vy'] = combined_agent_data['y'].diff().fillna(0)
                state_cols.extend(['vx', 'vy'])
                
                # Extract and clean state matrix
                X_combined = combined_agent_data[state_cols].values
                X_combined = pd.DataFrame(X_combined).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
                
                # Print overall statistics
                print(f"  Combined trajectory statistics:")
                for i, col in enumerate(state_cols):
                    col_data = X_combined[:, i]
                    print(f"    {col}: mean={np.mean(col_data):.3f}, std={np.std(col_data):.3f}, range=[{np.min(col_data):.3f}, {np.max(col_data):.3f}]")
                
                # Don't normalize at all - keep raw values for better dynamics discovery
                X_final = X_combined
                
                # Create continuous time vector
                t_combined = combined_agent_data['frame_continuous'].values.astype(np.float64)
                
                # Store as single long trajectory
                all_X = [X_final]
                all_t = [t_combined]
                trajectory_info = [{
                    'combined_simulations': len(all_sim_data),
                    'total_length': len(X_final),
                    'simulation_ids': [data['simulation_id'].iloc[0] for data in all_sim_data]
                }]
            
            else:
                # Single simulation case
                agent_sim = agent_sim.sort_values('frame')
                
                # Single simulation case
                agent_sim = agent_sim.sort_values('frame')
                
                state_cols = ['x', 'y']
                
                # Add velocity for richer dynamics
                agent_sim['vx'] = agent_sim['x'].diff().fillna(0)
                agent_sim['vy'] = agent_sim['y'].diff().fillna(0)
                state_cols.extend(['vx', 'vy'])
                
                X_sim = agent_sim[state_cols].values
                X_sim = pd.DataFrame(X_sim).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
                
                # Don't normalize - keep raw values
                X_final = X_sim
                
                # Create time vector from frames
                t_sim = (agent_sim['frame'] - agent_sim['frame'].min()).values.astype(np.float64)
                
                print(f"  Single trajectory statistics:")
                for i, col in enumerate(state_cols):
                    col_data = X_final[:, i]
                    print(f"    {col}: mean={np.mean(col_data):.3f}, std={np.std(col_data):.3f}, range=[{np.min(col_data):.3f}, {np.max(col_data):.3f}]")
                
                all_X = [X_final]
                all_t = [t_sim]
                trajectory_info = [{'simulation_id': 'single', 'length': len(X_final)}]
            
            # Store agent data with trajectory information
            if all_X:  # Only if we have valid trajectories
                agent_data.append({
                    'agent_id': agent_id,
                    'trajectories': {
                        'X': all_X,
                        't': all_t,
                        'info': trajectory_info
                    },
                    'state_names': state_cols,
                    'num_trajectories': len(all_X)
                })
                
                if trajectory_info[0].get('combined_simulations'):
                    print(f"  Agent {agent_id}: 1 combined trajectory from {trajectory_info[0]['combined_simulations']} simulations, total length: {len(all_X[0])}")
                else:
                    print(f"  Agent {agent_id}: 1 trajectory, length: {len(all_X[0])}")
        
        return agent_data
    
    def smooth_trajectories(self, X: np.ndarray, window_length: int = 5, 
                          polyorder: int = 2) -> np.ndarray:
        """
        Smooth state trajectories using Savitzky-Golay filter.
        
        Args:
            X: State matrix
            window_length: Window length for smoothing
            polyorder: Polynomial order for smoothing
            
        Returns:
            Smoothed state matrix
        """
        if len(X) < window_length:
            return X
            
        X_smooth = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_smooth[:, i] = savgol_filter(X[:, i], window_length, polyorder)
        
        return X_smooth


class SINDyAnalyzer:
    """Apply SINDy analysis to discover governing equations."""
    
    def __init__(self):
        self.models = {}
        
    def fit_sindy_model(self, trajectories: Dict, state_names: List[str],
                       agent_id: str, threshold: float = 0.1,
                       degree: int = 2) -> Optional[object]:
        """
        Fit SINDy model to agent data with multiple trajectories.
        
        Args:
            trajectories: Dictionary containing X, t lists of trajectories
            state_names: Names of state variables  
            agent_id: Agent identifier
            threshold: Sparsity threshold for STLSQ
            degree: Polynomial degree for feature library
            
        Returns:
            Fitted SINDy model or None if not available
        """
        if not SINDY_AVAILABLE:
            print("SINDy not available. Please install pysindy.")
            return None
            
        try:
            X_list = trajectories['X']
            t_list = trajectories['t']
            
            if not X_list:
                print(f"No trajectories available for agent {agent_id}")
                return None
            
            # Clean and validate each trajectory
            clean_X = []
            clean_t = []
            
            for i, (X, t) in enumerate(zip(X_list, t_list)):
                # Ensure proper arrays
                X = np.asarray(X, dtype=np.float64)
                t = np.asarray(t, dtype=np.float64)
                
                # Clean data
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
                
                if np.any(np.isnan(t)) or np.any(np.isinf(t)):
                    t = np.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Check if trajectory is meaningful (has variation)
                if X.shape[0] > 5 and np.std(X) > 1e-6:
                    clean_X.append(X)
                    clean_t.append(t)
                else:
                    print(f"  Skipping trajectory {i} for agent {agent_id}: insufficient variation")
            
            if not clean_X:
                print(f"No valid trajectories found for agent {agent_id}")
                return None
            
            print(f"  Using {len(clean_X)} clean trajectories for agent {agent_id}")
            
            # Create feature library with reasonable degree
            feature_library = ps.PolynomialLibrary(degree=min(degree, 2))  # Use degree 2 for richer dynamics
            
            # Create optimizer with adjusted threshold for better results
            optimizer = ps.STLSQ(threshold=max(threshold, 0.01), alpha=0.01)  # Ensure minimum threshold
            
            # Create SINDy model for autonomous system (no control inputs)
            model = ps.SINDy(
                feature_library=feature_library,
                optimizer=optimizer,
                discrete_time=False
            )
            
            # Fit model with multiple trajectories (autonomous system)
            try:
                # For multiple trajectories, use the first few trajectories to avoid memory issues
                if len(clean_X) > 5:
                    # Use first 5 trajectories for fitting to avoid complexity
                    X_subset = clean_X[:5]
                    t_subset = clean_t[:5]
                    print(f"  Using first 5 trajectories out of {len(clean_X)} for fitting")
                else:
                    X_subset = clean_X
                    t_subset = clean_t
                
                # Concatenate trajectories without NaN separators
                X_concat = np.vstack(X_subset)
                t_parts = []
                t_offset = 0
                for t_traj in t_subset:
                    t_parts.append(t_traj + t_offset)
                    t_offset += len(t_traj)  # Ensure continuity
                t_concat = np.concatenate(t_parts)
                
                # Use actual time differences
                if len(t_concat) > 1:
                    dt = np.median(np.diff(t_concat))
                    if dt <= 0 or np.isnan(dt):
                        dt = 1.0
                else:
                    dt = 1.0
                
                print(f"  Using dt = {dt:.3f}")
                model.fit(X_concat, t=dt)
                
                # Score on first clean trajectory
                score = model.score(clean_X[0], t=dt)
                
            except Exception as e:
                print(f"  Failed to fit model: {e}")
                return None
            
            # Store model with trajectory information
            self.models[agent_id] = {
                'model': model,
                'trajectories': {
                    'X': clean_X,
                    't': clean_t
                },
                'X': clean_X[0],  # Keep first trajectory for compatibility
                't': clean_t[0],
                'state_names': state_names,
                'score': score,
                'num_trajectories': 1  # Now always 1 since we combine everything
            }
            
            print(f"  SINDy model fitted for agent {agent_id}")
            print(f"  Model score: {score:.4f}")
            if 'combined_simulations' in trajectories['info'][0]:
                print(f"  Used 1 combined trajectory from {trajectories['info'][0]['combined_simulations']} simulations")
            else:
                print(f"  Used 1 trajectory")
            
            return model
            
        except Exception as e:
            print(f"Error fitting SINDy model for agent {agent_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_equations(self, agent_id: str) -> None:
        """Print the discovered equations for an agent."""
        if agent_id in self.models:
            print(f"\nDiscovered equations for Agent {agent_id}:")
            print("=" * 50)
            self.models[agent_id]['model'].print()
        else:
            print(f"No model found for agent {agent_id}")
    
    def save_model_results(self, agent_id: str, output_dir: Path) -> None:
        """Save model results (equations and coefficients) to files."""
        if agent_id not in self.models:
            print(f"No model found for agent {agent_id}")
            return
            
        model_data = self.models[agent_id]
        model = model_data['model']
        
        try:
            # Create agent-specific directory
            agent_dir = output_dir / f"agent_{agent_id}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            
            # Save equations to text file
            equations_file = agent_dir / "equations.txt"
            with open(equations_file, 'w') as f:
                f.write(f"SINDy Equations for Agent {agent_id}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Model Score: {model_data['score']:.6f}\n\n")
                f.write("State Variables: " + ", ".join(model_data['state_names']) + "\n\n")
                f.write("Discovered Equations:\n")
                f.write("-" * 30 + "\n")
                
                # Capture model.print() output
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                model.print()
                equations_text = buffer.getvalue()
                sys.stdout = old_stdout
                
                f.write(equations_text)
            
            # Save coefficients to CSV
            coefficients_file = agent_dir / "coefficients.csv"
            try:
                # Try to get feature names (newer versions)
                feature_names = model.get_feature_names()
            except AttributeError:
                # Fallback for older versions or when method doesn't exist
                feature_names = [f"feature_{i}" for i in range(model.coefficients().shape[1])]
            
            coeffs_df = pd.DataFrame(
                model.coefficients(),
                columns=feature_names,
                index=model_data['state_names']
            )
            coeffs_df.to_csv(coefficients_file)
            
            # Save model metadata
            metadata_file = agent_dir / "metadata.json"
            import json
            metadata = {
                'agent_id': agent_id,
                'model_score': float(model_data['score']),
                'state_names': model_data['state_names'],
                'data_shape': {
                    'X_shape': list(model_data['X'].shape),
                    't_length': len(model_data['t'])
                }
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model results saved for agent {agent_id} in {agent_dir}")
            
        except Exception as e:
            print(f"Error saving model results for agent {agent_id}: {e}")
    
    def plot_model_comparison(self, agent_id: str, output_dir: Path, save_plots: bool = True) -> None:
        """Plot comparison between original and predicted trajectories."""
        if agent_id not in self.models:
            print(f"No model found for agent {agent_id}")
            return
            
        model_data = self.models[agent_id]
        model = model_data['model']
        
        try:
            # Plot first trajectory for comparison
            X = model_data['X']  # First trajectory
            t = model_data['t']
            state_names = model_data['state_names']
            
            # Predict using the model
            try:
                # Predict trajectories (autonomous system)
                X_pred = model.predict(X)
                    
            except Exception as e:
                print(f"  Error in prediction: {e}")
                print(f"  Could not generate predictions for agent {agent_id}")
                return
            
            # Create subplots
            n_states = min(X.shape[1], len(state_names))
            fig, axes = plt.subplots(n_states, 1, figsize=(12, 2*n_states))
            if n_states == 1:
                axes = [axes]
            
            for i in range(n_states):
                ax = axes[i] if n_states > 1 else axes
                state_name = state_names[i] if i < len(state_names) else f'State {i}'
                
                ax.plot(t, X[:, i], 'b-', label='True', linewidth=2, alpha=0.8)
                if X_pred.shape[1] > i:
                    ax.plot(t, X_pred[:, i], 'r--', label='SINDy Predicted', linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Time (frames)')
                ax.set_ylabel(state_name)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{state_name} - Agent {agent_id}')
                
                # Add score information
                if i == 0 and 'trajectories' in model_data:
                    score_text = f"Model Score: {model_data['score']:.4f}\\nTrajectories: {model_data.get('num_trajectories', 1)}"
                    ax.text(0.02, 0.98, score_text, transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top', fontsize=10)
            
            plt.tight_layout()
            
            if save_plots:
                agent_dir = output_dir / f"agent_{agent_id}"
                agent_dir.mkdir(parents=True, exist_ok=True)
                plot_path = agent_dir / "trajectory_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  Plot saved to {plot_path}")
            
            plt.close()  # Close figure to save memory
            
        except Exception as e:
            print(f"Error plotting model comparison for agent {agent_id}: {e}")
            import traceback
            traceback.print_exc()


class SINDyPipeline:
    """Complete pipeline for SINDy analysis."""

    def __init__(self, cluster="cuenca"):
        """Initialize the pipeline."""
        self.extractor = CSVDataExtractor(cluster=cluster)
        self.preprocessor = SINDyPreprocessor()
        self.analyzer = SINDyAnalyzer()
        
    def run_analysis(self, map_nr: str, training_id: str, checkpoint_number: str,
                    simulation_id: str = None, combine_simulations: bool = False,
                    smooth_data: bool = True, sindy_threshold: float = 0.1, 
                    polynomial_degree: int = 2, save_plots: bool = True, **kwargs) -> Dict:
        """
        Run complete SINDy analysis pipeline.
        
        Args:
            map_nr: Map number/name
            training_id: Training ID
            checkpoint_number: Checkpoint number
            simulation_id: Simulation ID
            smooth_data: Whether to smooth trajectories
            sindy_threshold: Sparsity threshold for SINDy
            polynomial_degree: Polynomial degree for feature library
            save_plots: Whether to save plots
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with analysis results
        """
        print("Starting SINDy Analysis Pipeline")
        print("=" * 50)
        
        # Step 1: Extract data
        print("\n1. Extracting CSV data...")
        try:
            if combine_simulations or simulation_id is None:
                print("Using combined data from all simulations...")
                data = self.extractor.extract_all_simulations_data(
                    map_nr, training_id, checkpoint_number, **kwargs
                )
            else:
                print(f"Using single simulation: {simulation_id}")
                data = self.extractor.extract_csv_data(
                    map_nr, training_id, checkpoint_number, simulation_id, **kwargs
                )
        except Exception as e:
            print(f"Error extracting data: {e}")
            return {}
        
        # Step 2: Align timeframes
        print("\n2. Aligning timeframes...")
        simulation_df, actions_df = self.preprocessor.align_timeframes(
            data['simulation'], data['actions']
        )
        
        # Step 3: Create state vectors
        print("\n3. Creating state vectors...")
        agent_data = self.preprocessor.create_state_vectors(simulation_df, actions_df)
        
        if not agent_data:
            print("No agent data created. Check your input files.")
            return {}
        
        # Step 4: Create output directory
        print("\n4. Creating output directory...")
        output_dir = data['metadata']['sindy_output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Step 5: Apply SINDy to each agent
        print("\n5. Applying SINDy analysis...")
        results = {'models': {}, 'metadata': data['metadata'], 'output_dir': str(output_dir)}
        
        for agent_info in agent_data:
            agent_id = agent_info['agent_id']
            trajectories = agent_info['trajectories']
            
            print(f"\nProcessing Agent {agent_id}:")
            info = trajectories['info'][0]
            if info.get('combined_simulations'):
                print(f"  Combined trajectory from {info['combined_simulations']} simulations")
                print(f"  Total length: {info['total_length']} points")
                print(f"  Simulation IDs: {', '.join(info['simulation_ids'][:5])}{'...' if len(info['simulation_ids']) > 5 else ''}")
            else:
                print(f"  Single trajectory: {info['length']} points")
            
            # Smooth data if requested
            if smooth_data and len(trajectories['X'][0]) > 5:
                print("  Applying smoothing to trajectory...")
                X_smooth = self.preprocessor.smooth_trajectories(trajectories['X'][0])
                trajectories['X'] = [X_smooth]
                print("  Applied smoothing to trajectory")
            
            # Fit SINDy model
            model = self.analyzer.fit_sindy_model(
                trajectories,
                agent_info['state_names'],
                str(agent_id),
                threshold=sindy_threshold,
                degree=polynomial_degree
            )
            
            if model is not None:
                results['models'][str(agent_id)] = self.analyzer.models[str(agent_id)]
                
                # Print equations
                self.analyzer.print_equations(str(agent_id))
                
                # Save model results
                self.analyzer.save_model_results(str(agent_id), output_dir)
                
                # Plot comparison
                self.analyzer.plot_model_comparison(str(agent_id), output_dir, save_plots)
        
        # Step 6: Save overall summary and create overview plots
        print("\n6. Saving analysis summary...")
        self._save_analysis_summary(results, output_dir)
        
        if save_plots and results['models']:
            print("\n7. Creating overview plots...")
            self._create_overview_plots(results, output_dir)
        
        print(f"\nAnalysis complete! Processed {len(results['models'])} agents.")
        print(f"All results saved in: {output_dir}")
        return results
    
    def _save_analysis_summary(self, results: Dict, output_dir: Path) -> None:
        """Save overall analysis summary."""
        try:
            import json
            from datetime import datetime
            
            # Convert PosixPath objects to strings for JSON serialization
            metadata_serializable = {}
            for key, value in results['metadata'].items():
                if isinstance(value, Path):
                    metadata_serializable[key] = str(value)
                else:
                    metadata_serializable[key] = value
            
            summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'metadata': metadata_serializable,
                'agents_processed': list(results['models'].keys()),
                'model_scores': {
                    agent_id: model_data['score'] 
                    for agent_id, model_data in results['models'].items()
                },
                'total_agents': len(results['models'])
            }
            
            summary_file = output_dir / "analysis_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Also save as readable text
            summary_text_file = output_dir / "analysis_summary.txt"
            with open(summary_text_file, 'w') as f:
                f.write("SINDy Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Date: {summary['analysis_timestamp']}\n")
                f.write(f"Map: {results['metadata']['map_nr']}\n")
                f.write(f"Training ID: {results['metadata']['training_id']}\n")
                f.write(f"Checkpoint: {results['metadata']['checkpoint_number']}\n")
                if 'simulation_ids' in results['metadata']:
                    f.write(f"Simulations Combined: {len(results['metadata']['simulation_ids'])}\n")
                    f.write(f"Simulation IDs: {', '.join(results['metadata']['simulation_ids'])}\n")
                else:
                    f.write(f"Simulation ID: {results['metadata']['simulation_id']}\n")
                f.write(f"Total Agents Processed: {len(results['models'])}\n\n")
                
                f.write("Model Scores by Agent:\n")
                f.write("-" * 30 + "\n")
                for agent_id, score in summary['model_scores'].items():
                    f.write(f"Agent {agent_id}: {score:.6f}\n")
            
            print(f"Analysis summary saved to {summary_file}")
            
        except Exception as e:
            print(f"Error saving analysis summary: {e}")
    
    def _create_overview_plots(self, results: Dict, output_dir: Path) -> None:
        """Create overview plots comparing all agents."""
        try:
            # Model scores comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Model scores bar chart
            agent_ids = list(results['models'].keys())
            scores = [results['models'][agent_id]['score'] for agent_id in agent_ids]
            
            ax1.bar(agent_ids, scores, color='skyblue', alpha=0.7)
            ax1.set_xlabel('Agent ID')
            ax1.set_ylabel('Model Score (R²)')
            ax1.set_title('SINDy Model Performance by Agent')
            ax1.grid(True, alpha=0.3)
            
            # Add score values on bars
            for i, (agent_id, score) in enumerate(zip(agent_ids, scores)):
                ax1.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
            
            # Plot 2: Trajectory lengths comparison
            trajectory_lengths = [len(results['models'][agent_id]['t']) for agent_id in agent_ids]
            
            ax2.bar(agent_ids, trajectory_lengths, color='lightcoral', alpha=0.7)
            ax2.set_xlabel('Agent ID')
            ax2.set_ylabel('Trajectory Length (frames)')
            ax2.set_title('Data Length by Agent')
            ax2.grid(True, alpha=0.3)
            
            # Add length values on bars
            for i, (agent_id, length) in enumerate(zip(agent_ids, trajectory_lengths)):
                ax2.text(i, length + max(trajectory_lengths)*0.01, str(length), ha='center', va='bottom')
            
            plt.tight_layout()
            overview_plot_path = output_dir / "overview_comparison.png"
            plt.savefig(overview_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Overview plots saved to {overview_plot_path}")
            
            # Create a summary table plot
            self._create_summary_table_plot(results, output_dir)
            
        except Exception as e:
            print(f"Error creating overview plots: {e}")
    
    def _create_summary_table_plot(self, results: Dict, output_dir: Path) -> None:
        """Create a summary table as a plot."""
        try:
            # Prepare data for table
            agent_ids = list(results['models'].keys())
            data_for_table = []
            
            for agent_id in agent_ids:
                model_data = results['models'][agent_id]
                data_for_table.append([
                    agent_id,
                    f"{model_data['score']:.4f}",
                    str(len(model_data['t'])),
                    f"{model_data['X'].shape[1]}"
                ])
            
            # Create table plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(agent_ids) * 0.4 + 2)))
            ax.axis('tight')
            ax.axis('off')
            
            columns = ['Agent ID', 'Model Score', 'Trajectory Length', 'State Vars']
            table = ax.table(cellText=data_for_table, colLabels=columns, 
                           cellLoc='center', loc='center')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color header
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(data_for_table) + 1):
                for j in range(len(columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
            
            plt.title('SINDy Analysis Summary Table', fontsize=14, fontweight='bold', pad=20)
            
            table_plot_path = output_dir / "summary_table.png"
            plt.savefig(table_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Summary table saved to {table_plot_path}")
            
        except Exception as e:
            print(f"Error creating summary table plot: {e}")


def main():
    """Main function to run the SINDy analysis pipeline."""
    parser = argparse.ArgumentParser(description='SINDy Analysis Pipeline for Spoiled Broth')
    
    # Required parameters
    parser.add_argument('--map_nr', type=str, required=True,
                       help='Map number/name (e.g., baseline_division_of_labor)')
    parser.add_argument('--training_id', type=str, required=True,
                       help='Training ID (e.g., 2025-09-13_13-27-52)')
    parser.add_argument('--checkpoint_number', type=str, required=True,
                       help='Checkpoint number (e.g., final, 50)')
    parser.add_argument('--simulation_id', type=str, default=None,
                       help='Simulation ID (e.g., 2025-09-13_13-27-52). If not provided, all simulations will be combined.')
    
    # Optional parameters
    parser.add_argument('--combine_simulations', action='store_true',
                       help='Combine all simulations for the given map/training/checkpoint')
    parser.add_argument('--cluster', type=str, default='cuenca',
                       help='Cluster name (cuenca, local)')
    parser.add_argument('--game_version', type=str, default='classic',
                       help='Game version (classic, competition)')
    parser.add_argument('--intent_version', type=str, default='v3.1',
                       help='Intent version')
    parser.add_argument('--cooperative', type=bool, default=True,
                       help='Whether cooperative (True) or competitive (False)')
    
    # SINDy parameters
    parser.add_argument('--smooth_data', type=bool, default=True,
                       help='Whether to smooth trajectories')
    parser.add_argument('--sindy_threshold', type=float, default=0.1,
                       help='Sparsity threshold for SINDy')
    parser.add_argument('--polynomial_degree', type=int, default=5,
                       help='Polynomial degree for feature library')
    parser.add_argument('--save_plots', type=bool, default=True,
                       help='Whether to save plots')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SINDyPipeline(args.cluster)
    
    # Run analysis
    results = pipeline.run_analysis(
        map_nr=args.map_nr,
        training_id=args.training_id,
        checkpoint_number=args.checkpoint_number,
        simulation_id=args.simulation_id,
        combine_simulations=args.combine_simulations,
        game_version=args.game_version,
        intent_version=args.intent_version,
        cooperative=args.cooperative,
        smooth_data=args.smooth_data,
        sindy_threshold=args.sindy_threshold,
        polynomial_degree=args.polynomial_degree,
        save_plots=args.save_plots
    )
    
    # Print summary
    if results:
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Map: {args.map_nr}")
        print(f"Training ID: {args.training_id}")
        print(f"Checkpoint: {args.checkpoint_number}")
        if 'simulation_ids' in results.get('metadata', {}):
            print(f"Combined {len(results['metadata']['simulation_ids'])} simulations")
        else:
            print(f"Simulation: {args.simulation_id}")
        print(f"Agents processed: {len(results.get('models', {}))}")
        
        for agent_id, model_data in results.get('models', {}).items():
            print(f"Agent {agent_id} - Model score: {model_data['score']:.4f}")


if __name__ == "__main__":
    main()