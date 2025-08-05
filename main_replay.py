import eventlet

eventlet.monkey_patch()
from engine.app.session_app import SessionApp
from engine.logging.replay_loader import load_replay_agents
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rl_controller import RLController
from spoiled_broth.llm.llm_controler import LLMController
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path
from spoiled_broth.config import *
import json

import logging

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path(__file__).resolve().parent / "spoiled_broth"

replay_path = "replays/replay.json"

with open(replay_path, "r") as f:
    replay_data = json.load(f)

config = replay_data["config"]
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
    tick_rate=12,
    ai_tick_rate=480,
    n_players=1,
    is_max_speed=False
)

app = engine_app.app

if __name__ == "__main__":
    import eventlet.wsgi

    engine_app.socketio.run(engine_app.app,
                            host="0.0.0.0",
                            port=8080,
                            debug=True)
