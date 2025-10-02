"""
Action tracking utilities for simulation runs.

Author: Samuel Lozano
"""

import time
import csv
import threading
from pathlib import Path
from typing import Any, Tuple, Optional


class ActionTracker:
    """Tracks agent actions with detailed logging and timing."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        
        # Track active actions (no longer need completion times for minimum duration)
        self.active_actions = {}
        self.action_completion_times = {}  # Keep for compatibility but not used for duration enforcement
        
        # Action counters per agent (for action_number)
        self.action_counters = {}
        
        # Action ID counters per agent (for action_id - 0-indexed per agent)
        self.action_id_counters = {}
        
        # Store completed actions for writing to CSV
        self.completed_actions = []
        
        self._initialize_csv()
    
    @staticmethod
    def _position_to_tile(x: float, y: float) -> tuple[int, int]:
        """Convert pixel position to tile coordinates consistently."""
        # Use consistent tile calculation: floor division by tile size + 1
        # This ensures all components use the same calculation
        if x is None or y is None:
            return 0, 0
        tile_x = int(x // 16 + 1)
        tile_y = int(y // 16 + 1)
        return tile_x, tile_y
    
    def _initialize_csv(self):
        """Initialize the action tracking CSV file."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'action_id', 'agent_id', 'action_number', 'action_type', 
                'target_tile_type', 'target_tile_x', 'target_tile_y'
            ])
    
    def _get_tile_info(self, game: Any, target_id: str) -> Tuple[str, Optional[int], Optional[int]]:
        """Get tile information for a given target ID."""
        tile_type, x_tile, y_tile = '', None, None
        try:
            grid = getattr(game, 'grid', None)
            if grid is not None and hasattr(grid, 'tiles'):
                for x in range(grid.width):
                    for y in range(grid.height):
                        t = grid.tiles[x][y]
                        if t and getattr(t, 'id', None) == target_id:
                            tile_type = t.__class__.__name__.lower()
                            x_tile, y_tile = x + 1, y + 1
                            return tile_type, x_tile, y_tile
        except Exception:
            pass
        return tile_type, x_tile, y_tile
    
    def _extract_action_info(self, action: Any, game: Any) -> Tuple[str, str, str, Optional[int], Optional[int]]:
        """Extract action type and target information."""
        action_type = ''
        target_id = ''
        tile_type = ''
        x_tile = None
        y_tile = None
        
        try:
            if action is None:
                return 'do_nothing', '', '', None, None
            
            if isinstance(action, dict):
                action_type = str(action.get('type', action.get('action', 'unknown')))
                target_id = str(action.get('target', ''))
            elif hasattr(action, 'action_type') or hasattr(action, 'type'):
                action_type = str(getattr(action, 'action_type', getattr(action, 'type', 'unknown')))
                target_id = str(getattr(action, 'target', ''))
            else:
                action_type = str(action)
            
            if target_id:
                tile_type, x_tile, y_tile = self._get_tile_info(game, target_id)
                
        except Exception as e:
            action_type = 'unknown'
            print(f"DEBUG: Error extracting action info: {e}")
            
        return action_type, target_id, tile_type, x_tile, y_tile
    
    def start_action(self, agent_id: str, action_type: str, tile_or_target: Any, 
                    timestamp: float, action_number: Optional[int] = None, **kwargs):
        """Start tracking an action for an agent."""
        with self.lock:
            if agent_id in self.active_actions:
                # Log a warning if the agent is already performing an action
                print(f"[ACTION_TRACKER] Agent {agent_id} is already performing action "
                      f"'{self.active_actions[agent_id].get('action_type', 'unknown')}' and cannot start a new one.")
                return

            game = getattr(self, 'game', None)
            engine = getattr(self, 'engine', None)
            
            if game is None:
                print(f"[ACTION_TRACKER] No game reference for {agent_id}")
                return
            
            agent_obj = game.gameObjects.get(agent_id)
            
            if agent_obj is None:
                print(f"[ACTION_TRACKER] No agent object found for {agent_id}")
                return
            
            # Create action dict for processing
            if action_type == "click" and tile_or_target:
                action = {"type": "click", "target": getattr(tile_or_target, 'id', str(tile_or_target))}
            elif action_type == "do_nothing":
                action = None
            else:
                action = {"type": action_type}
            
            # Extract action information
            action_type_str, target_id, tile_type, x_tile, y_tile = self._extract_action_info(action, game)
            
            # Get agent current state at action start time
            agent_x = getattr(agent_obj, 'x', None)
            agent_y = getattr(agent_obj, 'y', None)
            # Capture item state BEFORE the action starts
            item_before = getattr(agent_obj, 'item', None)
            
            # Use provided action number or generate new one
            if action_type == "do_nothing":
                # do_nothing actions should always have action_number = -1
                action_number = -1
            elif action_number is None or action_number == -1:
                # Generate new action number for regular actions
                self.action_counters[agent_id] = self.action_counters.get(agent_id, 0) + 1
                action_number = self.action_counters[agent_id]
            
            # Create action record
            action_record = {
                'action': action,
                'action_type': action_type_str,
                'target_id': target_id,
                'tile_type': tile_type,
                'target_x_tile': x_tile,
                'target_y_tile': y_tile,
                'decision_timestamp': timestamp,
                'decision_x': agent_x,
                'decision_y': agent_y,
                'item_before': item_before,
                'action_number': action_number
            }
            
            # Store active action
            self.active_actions[agent_id] = action_record
            
            # No minimum duration enforcement - actions complete when the game engine says they're complete
            # Remove the action from completion times tracking
            self.action_completion_times.pop(agent_id, None)
                
    # Removed can_complete_action method - actions complete when game engine says they're complete
    
    def end_action(self, agent_id: str, timestamp: Optional[float] = None, current_frame: Optional[int] = None, **kwargs) -> bool:
        """End tracking an action for an agent."""
        with self.lock:
            if agent_id not in self.active_actions:
                return False
            
            action_record = self.active_actions[agent_id]
            game = getattr(self, 'game', None)
            engine = getattr(self, 'engine', None)
            
            if game is None or engine is None:
                self.active_actions.pop(agent_id, None)
                self.action_completion_times.pop(agent_id, None)
                return False
            
            # Use current timestamp if provided, otherwise get current time
            if timestamp is None:
                timestamp = time.time()
            
            # Actions complete when the game engine says they're complete - no duration checks
            
            # Get action_id for this agent (0-indexed per agent)
            if agent_id not in self.action_id_counters:
                self.action_id_counters[agent_id] = 0
            else:
                self.action_id_counters[agent_id] += 1
            
            action_id = self.action_id_counters[agent_id]
            
            # Write action immediately to CSV in simplified format
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [
                    action_id,
                    agent_id,
                    action_record['action_number'],
                    action_record['action_type'],
                    action_record['tile_type'] or '',
                    action_record['target_x_tile'] or '',
                    action_record['target_y_tile'] or ''
                ]
                writer.writerow(row)
                        
            # Remove from active tracking
            self.active_actions.pop(agent_id, None)
            self.action_completion_times.pop(agent_id, None)
            return True
    
    def finalize_actions_with_log_data(self, log_csv_path: Path, tick_rate: int = 24):
        """
        Legacy method kept for compatibility - actions are now written directly during end_action.
        """
        print("[ACTION_TRACKER] Actions are now written directly to CSV - no finalization needed")
        pass
    

    def cleanup(self, game: Any, engine: Any, max_cleanup_time: float = 3.0):
        """Complete any remaining active actions at simulation end."""
        print(f"[ACTION_TRACKER] Starting cleanup for {len(self.active_actions)} active actions")
        
        cleanup_start_time = time.time()
        
        try:
            lock_acquired = self.lock.acquire(timeout=2.0)
            if not lock_acquired:
                print(f"[ACTION_TRACKER] Could not acquire lock for cleanup")
                self.active_actions.clear()
                self.action_completion_times.clear()
                return
            
            try:
                active_agents = list(self.active_actions.keys())
                for agent_id in active_agents:
                    if time.time() - cleanup_start_time > max_cleanup_time:
                        print(f"[ACTION_TRACKER] Cleanup timeout reached")
                        self.active_actions.clear()
                        self.action_completion_times.clear()
                        break
                    
                try:
                    # At simulation end, complete any remaining actions with final timestamp
                    final_timestamp = time.time()
                    print(f"[ACTION_TRACKER] Completing remaining action for {agent_id} at simulation end")
                    self._force_complete_action(agent_id, game, engine, final_timestamp)
                except Exception as action_error:
                    print(f"[ACTION_TRACKER] Error completing action for {agent_id}: {action_error}")
                    # Manually remove from tracking if completion fails
                    self.active_actions.pop(agent_id, None)
                    self.action_completion_times.pop(agent_id, None)
                
            finally:
                self.lock.release()
                
        except Exception as e:
            print(f"[ACTION_TRACKER] Critical error during cleanup: {e}")
            self.active_actions.clear()
            self.action_completion_times.clear()
    
    def _force_complete_action(self, agent_id: str, game: Any, engine: Any, final_timestamp: Optional[float] = None):
        """Force complete an action during cleanup."""
        if agent_id not in self.active_actions:
            return
        
        action_record = self.active_actions[agent_id]
        
        # Use provided final_timestamp or get current time
        if final_timestamp is not None:
            completion_timestamp = final_timestamp
        else:
            completion_timestamp = time.time()
        
        agent_obj = game.gameObjects.get(agent_id) if game else None
        
        # Get completion state
        if agent_obj:
            agent_x = getattr(agent_obj, 'x', action_record['decision_x'] or 0)
            agent_y = getattr(agent_obj, 'y', action_record['decision_y'] or 0)
            item_after = getattr(agent_obj, 'item', None)
        else:
            agent_x = action_record['decision_x'] or 0
            agent_y = action_record['decision_y'] or 0  
            item_after = action_record['item_before']
        
        # Convert positions to tile coordinates
        decision_tile_x, decision_tile_y = self._position_to_tile(
            action_record['decision_x'], action_record['decision_y']
        )
        completion_tile_x, completion_tile_y = self._position_to_tile(agent_x, agent_y)
        
        # Store completed action data WITHOUT frame numbers (to be added later)
        completed_action = {
            'agent_id': agent_id,
            'action_number': action_record['action_number'],
            'action_type': action_record['action_type'],
            'target_tile_type': action_record.get('tile_type') or 'None',
            'target_tile_x': action_record.get('target_x_tile') or 0,
            'target_tile_y': action_record.get('target_y_tile') or 0,
            'decision_x': action_record['decision_x'],
            'decision_y': action_record['decision_y'],
            'decision_tile_x': decision_tile_x,
            'decision_tile_y': decision_tile_y,
            'completion_x': agent_x,
            'completion_y': agent_y,
            'completion_tile_x': completion_tile_x,
            'completion_tile_y': completion_tile_y,
            'item_before': action_record['item_before'],
            'item_after': item_after,
            'decision_timestamp': action_record['decision_timestamp'],
            'completion_timestamp': completion_timestamp
        }
        
        # Add to completed actions list
        self.completed_actions.append(completed_action)
        
        # Remove from tracking
        self.active_actions.pop(agent_id, None)
        self.action_completion_times.pop(agent_id, None)