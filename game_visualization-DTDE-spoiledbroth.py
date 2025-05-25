from engine.app.session_app import SessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path
from spoiled_broth.config import *
from functools import partial
from spoiled_broth.rl.game_env import GameEnv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
import ray
from ray.tune.registry import register_env
from spoiled_broth.game import SpoiledBroth
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import os
import logging

map_nr = 1
training_id = 'PPO_spoiled_broth_2025-05-19_01-14-56m4dd6v31'
checkpoint_number = 1750

base_path = Path(f"/data/samuel_lozano/cooked/saved_models/map_{map_nr}/{training_id}")
config_path = base_path / "config.txt"
if config_path.exists():
    print(f"\n--- Contents of config.txt for training_id '{training_id}' ---")
    with open(config_path, "r") as f:
        for i in range(7):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())
else:
    print(f"config.txt not found at: {config_path}")

print(f"\n--- Using checkpoint number: {checkpoint_number} ---\n")

def env_creator(config):
    return GameEnv(**config)

# Ruta personalizada con espacio suficiente
custom_tmp_dir = Path("/data/samuel_lozano/tmp_ray")
custom_tmp_dir.mkdir(parents=True, exist_ok=True)
os.environ["RAY_TMPDIR"] = str(custom_tmp_dir)

# Inicializa Ray manualmente ANTES de que RLlib lo haga automáticamente
ray.shutdown()
ray.init(local_mode=True, _temp_dir=str(custom_tmp_dir), ignore_reinit_error=True, include_dashboard=False)

register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator(config)))

cf_AI_TICK_RATE = 2 # 2 decisions per second

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path().resolve() / "spoiled_broth"
checkpoint_dir = Path(f"/data/samuel_lozano/cooked/saved_models/map_{map_nr}/{training_id}/Checkpoint_{checkpoint_number}/")
if not checkpoint_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

engine_app = SessionApp(
    game_factory=partial(Game, map_nr=map_nr),
    ui_modules=[Renderer2DModule()],
    agent_map={
        "ai_1": RLlibController("ai_1", checkpoint_dir, "policy_ai_rl_1"),
        "ai_2": RLlibController("ai_2", checkpoint_dir, "policy_ai_rl_2")
    },
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=cf_AI_TICK_RATE,
    n_players=0, 
    is_max_speed=False
)

app = engine_app.app

if __name__ == "__main__":
    app.run(port=5000)