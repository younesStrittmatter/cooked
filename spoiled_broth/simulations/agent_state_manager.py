"""
Agent state manager for handling stuck agents in simulations.

This module provides utilities to monitor and reset agent states that get stuck
during simulation execution.
"""

import time
from typing import Dict, Any


class AgentStateManager:
    """Manages agent states to prevent agents from getting stuck."""
    
    def __init__(self):
        self.agent_state_history: Dict[str, Dict[str, Any]] = {}
        self.last_action_check = 0.0
        self.check_interval = 5.0  # Check every 5 seconds
        
    def register_agent(self, agent_id: str, agent):
        """Register an agent for state monitoring."""
        self.agent_state_history[agent_id] = {
            'agent': agent,
            'last_action_time': time.time(),
            'stuck_count': 0,
            'last_position': None
        }
    
    def update_agent_action(self, agent_id: str):
        """Mark that an agent has performed an action."""
        if agent_id in self.agent_state_history:
            self.agent_state_history[agent_id]['last_action_time'] = time.time()
            self.agent_state_history[agent_id]['stuck_count'] = 0
    
    def check_and_reset_stuck_agents(self) -> int:
        """Check for stuck agents and reset their states. Returns number of agents reset."""
        current_time = time.time()
        
        # Only check periodically
        if current_time - self.last_action_check < self.check_interval:
            return 0
            
        self.last_action_check = current_time
        reset_count = 0
        
        for agent_id, state in self.agent_state_history.items():
            agent = state['agent']
            last_action_time = state['last_action_time']
            
            # Check if agent has been stuck for too long
            if current_time - last_action_time > 10.0:  # 10 seconds without action
                is_busy = getattr(agent, 'is_busy', False)
                is_moving = getattr(agent, 'is_moving', False)
                
                if is_busy and not is_moving:
                    # Agent is marked as busy but not actually moving - reset
                    print(f"[AGENT_STATE_MANAGER] Resetting stuck agent {agent_id}")
                    agent.is_busy = False
                    if hasattr(agent, 'current_action'):
                        agent.current_action = None
                    if hasattr(agent, 'is_cutting'):
                        agent.is_cutting = False
                    if hasattr(agent, 'cutting_start_time'):
                        agent.cutting_start_time = None
                    
                    state['stuck_count'] += 1
                    state['last_action_time'] = current_time
                    reset_count += 1
        
        return reset_count