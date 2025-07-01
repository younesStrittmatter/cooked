from engine.app.session_app import SessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path
from spoiled_broth.config import *
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
import ray
from ray.tune.registry import register_env
from spoiled_broth.rl.game_env import GameEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

import logging

log = logging.getLogger('werkzeug')
log.disabled = True

# Configuration
map_nr = 1
training_id = 'PPO_spoiled_broth_2025-05-19_22-49-57bnxwywxn'
checkpoint_number = 450
local = '/mnt/lustre/home/samuloza'
base_path = Path(f"{local}/data/samuel_lozano/cooked/saved_models/map_{map_nr}/{training_id}")
checkpoint_dir = base_path / f"Checkpoint_{checkpoint_number}/"

if not checkpoint_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

def env_creator(config):
    return GameEnv(**config)

# Initialize Ray
ray.shutdown()
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator(config)))

path_root = Path(__file__).resolve().parent / "spoiled_broth"

engine_app = SessionApp(
    game_factory=partial(Game, map_nr=map_nr),
    ui_modules=[Renderer2DModule()],
    agent_map={
        "ai_1": RLlibController("ai_1", checkpoint_dir, "policy_ai_rl_1"),
        "ai_2": RLlibController("ai_2", checkpoint_dir, "policy_ai_rl_2")
    },
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=2,  # 2 actions per second
    n_players=0,  # No human players needed
    is_max_speed=False
)

app = engine_app.app

# Manually start the session since n_players=0
if engine_app.session_manager.sessions:
    session = list(engine_app.session_manager.sessions.values())[0]
    session.start()
    print("Session started manually for AI-only gameplay")
else:
    session = engine_app.session_manager.find_or_create_session()
    session.start()
    print("New session created and started for AI-only gameplay")

if __name__ == "__main__":
    app.run(port=5000)
