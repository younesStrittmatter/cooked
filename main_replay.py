import eventlet

eventlet.monkey_patch()
from engine.app.session_app import SessionApp
from engine.logging.replay_loader import load_replay_agents
from spoiled_broth.game import SpoiledBroth as Game
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path
import json
from spoiled_broth_experiment_settings.params import params_both, params_replay

import logging

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path(__file__).resolve().parent / "spoiled_broth"


def get_next_replay_path():
    replay_folder = './analysis/replays'
    import os
    for filename in os.listdir(replay_folder):
        if filename.endswith(".json"):
            csv_file = f'analysis/tick_logs/{filename.split(".")[0]}.csv'
            if not Path(csv_file).exists():
                return replay_folder + '/' + filename
    return None


replay_path = get_next_replay_path()
print('Using replay:', replay_path)

# replay_path = "replays/replay.json"

with open(replay_path, "r") as f:
    print(replay_path)
    replay_data = json.load(f)

config = replay_data["config"]
config["tick_log_path"] = f'analysis/tick_logs/{replay_path.split("/")[-1].split(".")[0]}.csv'

replay_agents = load_replay_agents(replay_path)


# Recreate the game from state
def create_game_from_config(url_params=None):
    game = Game.from_state(config)
    return game


restored_game = Game.from_state(config)

engine_app = SessionApp(
    game_factory=create_game_from_config,
    ui_modules=[Renderer2DModule()],
    agent_map=replay_agents,
    path_root=path_root,
    tick_rate=params_both['tick_rate'],
    ai_tick_rate=480,
    n_players=params_replay['n_players'],
    is_max_speed=params_replay['is_max_speed'],
    max_game_time=params_both['max_game_time'],
)

app = engine_app.app

if __name__ == "__main__":
    import eventlet.wsgi

    engine_app.socketio.run(engine_app.app,
                            host="0.0.0.0",
                            port=8080,
                            debug=True)
