"""
Game management utilities for simulation runs.

Author: Samuel Lozano
"""

from typing import Tuple, Callable, Any

from .simulation_config import SimulationConfig


class GameManager:
    """Manages game instance creation and configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def create_game_factory(self, map_nr: str, grid_size: Tuple[int, int], 
                           intent_version: str) -> Callable:
        """
        Create a game factory function for the simulation.
        
        Args:
            map_nr: Name of the map to load
            grid_size: Grid size tuple (width, height)
            intent_version: Intent version identifier
            
        Returns:
            Game factory function
        """
        def game_factory():
            from spoiled_broth.game import SpoiledBroth as Game
            
            game = Game(
                map_nr=map_nr, 
                grid_size=grid_size, 
                intent_version=intent_version
            )
            
            # Reset game state
            self._reset_game_state(game)
            return game
        
        return game_factory
    
    def _reset_game_state(self, game: Any):
        """Reset game state to clean initial conditions."""
        try:
            grid = getattr(game, 'grid', None)
            if grid is not None:
                from spoiled_broth.world.tiles import Dispenser
                
                # Clear items on non-dispenser tiles
                for x in range(grid.width):
                    for y in range(grid.height):
                        tile = grid.tiles[x][y]
                        if tile is None:
                            continue
                        
                        if hasattr(tile, 'item') and not isinstance(tile, Dispenser):
                            tile.item = None
                        # Reset cutting board state by resetting cut_time_accumulated
                        # (cut_stage is a read-only property calculated from cut_time_accumulated)
                        if hasattr(tile, 'cut_time_accumulated'):
                            tile.cut_time_accumulated = 0
                        if hasattr(tile, 'cut_by'):
                            tile.cut_by = None
                        if hasattr(tile, 'cut_item'):
                            tile.cut_item = None
            
            # Reset agent states
            for agent_id, obj in list(game.gameObjects.items()):
                if agent_id.startswith('ai_rl_'):
                    if hasattr(obj, 'item'):
                        obj.item = None
                    if hasattr(obj, 'action_complete'):
                        obj.action_complete = True
                    if hasattr(obj, 'current_action'):
                        obj.current_action = None
                    if hasattr(obj, 'speed'):
                        obj.speed = self.config.agent_speed_px_per_sec
                        
        except Exception as e:
            print(f"Warning: Error resetting game state: {e}")