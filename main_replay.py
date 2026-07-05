import eventlet

eventlet.monkey_patch()
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT / "engine" / "src", REPO_ROOT / "games", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engine.app.session_app import SessionApp
from engine.logging.replay_loader import load_replay_agents
from spoiled_broth.game import SpoiledBroth as Game
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
import json
from spoiled_broth_experiment_settings.params import params_both, params_replay

import logging

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = REPO_ROOT / "games/spoiled_broth"


def get_next_replay_path(filename=None):
    replay_folder = './analysis/replays'
    # replay_folder = './replays'
    import os
    if not filename:
        checked = 0
        for filename in sorted(os.listdir(replay_folder)):
            if filename.endswith(".json"):
                checked += 1
                csv_file = f'analysis/tick_logs/{filename.split(".")[0]}.csv'
                if not Path(csv_file).exists():
                    print(f"Checked {checked} replay files before finding pending work.", flush=True)
                    return replay_folder + '/' + filename
    else:
        csv_file = f'analysis/tick_logs/{filename.split(".")[0]}.csv'
        if Path(csv_file).exists():
            print(f"CSV file {csv_file} already exists. Skipping replay {filename}.")
            return None
        return replay_folder + '/' + filename
    return None


Path("analysis/tick_logs").mkdir(parents=True, exist_ok=True)

replay_path = get_next_replay_path()
if replay_path is None:
    print("No pending replay JSON found.")
    raise SystemExit(0)
print(f"Selected replay path: {replay_path}", flush=True)
try:
    with open(replay_path, "r") as f:
        replay_data = json.load(f)
except FileNotFoundError:
    print(replay_path, "not found")

config = replay_data["config"]
config["tick_log_path"] = f'analysis/tick_logs/{replay_path.split("/")[-1].split(".")[0]}.csv'
print(f"Writing tick log to: {config['tick_log_path']}", flush=True)

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
    ai_tick_rate=24,
    n_players=params_replay['n_players'],
    is_max_speed=params_replay['is_max_speed'],
    max_game_time=params_both['max_game_time'],
    is_served_locally=True,
    
)

app = engine_app.app


def run_server():
    import eventlet.wsgi
    engine_app.socketio.run(engine_app.app,
                            host="0.0.0.0",
                            port=8080,
                            debug=False, use_reloader=False)


if __name__ == "__main__":
    run_server()