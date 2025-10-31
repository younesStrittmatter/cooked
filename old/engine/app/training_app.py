# engine/app/training_app.py
import time
from engine.core import Engine


class TrainingApp:
    def __init__(self, game_factory, agent_map, tick_rate=20, max_ticks=1000):
        self.game_factory = game_factory
        self.agent_map = agent_map
        self.tick_rate = tick_rate
        self.max_ticks = max_ticks

    def run(self):
        game = self.game_factory()
        engine = Engine(game, tick_rate=self.tick_rate)

        for _ in range(self.max_ticks):
            if game.done:
                break

            observations = game.get_observations()
            for agent_id, agent in self.agent_map.items():
                obs = observations.get(agent_id)
                if obs is not None:
                    action = agent.choose_action(obs)
                    engine.submit_intent(agent_id, action)

            engine.tick()
            time.sleep(1.0 / self.tick_rate)

        return game
