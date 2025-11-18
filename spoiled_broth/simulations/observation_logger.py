"""
Observation space logging utilities for capturing observation vectors associated with agent actions.

This module provides logging capabilities to capture the observation vectors that agents
use when making action decisions, which is useful for debugging and analysis.

Author: Samuel Lozano
"""

import time
import csv
import threading
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union, List


class ObservationLogger:
    """Logs observation vectors associated with each agent action for debugging purposes."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        
        # Action ID counters per agent (for action_id - 0-indexed per agent)
        self.action_id_counters = {}
        
        # Track whether we've written the header (dynamic based on first observation)
        self.header_written = False
        self.observation_size = None
        
    def _initialize_csv(self, observation_vector: np.ndarray):
        """Initialize the observation logging CSV file with dynamic header based on observation size."""
        if self.header_written:
            return
            
        self.observation_size = len(observation_vector)
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Base columns
            header = ['action_id', 'agent_id', 'frame', 'action_type', 'action_name']
            
            # Add observation vector columns
            for i in range(self.observation_size):
                header.append(f'obs_{i}')
                
            writer.writerow(header)
            
        self.header_written = True
    
    def log_observation(self, agent_id: str, frame: int, action_type: str, 
                       action_name: str, observation_vector: np.ndarray):
        """
        Log an observation vector associated with an agent action.
        
        Args:
            agent_id: ID of the agent taking the action
            frame: Current game frame
            action_type: Type of action (e.g., 'click', 'do_nothing', 'rl_action')
            action_name: Name or description of the action
            observation_vector: The observation vector used for action selection
        """
        with self.lock:
            try:
                # Initialize CSV with header if this is the first observation
                if not self.header_written:
                    self._initialize_csv(observation_vector)
                
                # Validate observation vector size
                if len(observation_vector) != self.observation_size:
                    print(f"[OBS_LOGGER] Warning: Observation vector size mismatch for {agent_id}. "
                          f"Expected {self.observation_size}, got {len(observation_vector)}")
                    return
                
                # Get action_id for this agent (0-indexed per agent)
                if agent_id not in self.action_id_counters:
                    self.action_id_counters[agent_id] = 0
                else:
                    self.action_id_counters[agent_id] += 1
                
                action_id = self.action_id_counters[agent_id]
                
                # Write observation to CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Base row data
                    row = [
                        action_id,
                        agent_id,
                        frame,
                        action_type,
                        action_name
                    ]
                    
                    # Add observation vector values
                    row.extend(observation_vector.tolist())
                    
                    writer.writerow(row)
                                        
            except Exception as e:
                print(f"[OBS_LOGGER] Error logging observation for {agent_id}: {e}")
                import traceback
                traceback.print_exc()

    def get_observation_info(self) -> dict:
        """
        Get information about the logged observations.
        
        Returns:
            Dictionary with observation info including size and agent counts
        """
        info = {
            'observation_size': self.observation_size,
            'agents_logged': list(self.action_id_counters.keys()),
            'total_actions_per_agent': dict(self.action_id_counters),
            'header_written': self.header_written
        }
        return info