from engine.extensions.topDownGridWorld.ai_controller._base_controller import Controller
from stable_baselines3 import PPO

from spoiled_broth.game import game_to_vector

from pathlib import Path


class RLController(Controller):
    def __init__(self, agent_id, path="ppo_spoiled_broth.zip"):
        super().__init__(agent_id)
        model_path = Path(__file__).parent / "saved_models" / path

        self.model = PPO.load(model_path)
        self.agent_id = agent_id

    def choose_action(self, observation: dict):
        # agent will be assigned by SessionApp, so we can access the game through it
        obs_vector = game_to_vector(self.agent.game, self.agent_id)
        action, _ = self.model.predict(obs_vector, deterministic=True)

        grid = self.agent.grid
        x = action % grid.width
        y = action // grid.width
        tile = grid.tiles[x][y]
        print(f'chosen tile: {tile.slot_x, tile.slot_x}')

        if tile and hasattr(tile, "click"):
            return {"type": "click", "target": tile.id}
        return None
