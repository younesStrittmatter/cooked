import ray
from engine.app.ai_only_app import AIOnlySessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
from spoiled_broth.rl.rllib_controller_lstm import RLlibControllerLSTM
from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.game_env_competition import GameEnvCompetition
from engine.extensions.renderer2d.renderer_2d_offline import Renderer2DOffline
from pathlib import Path
from spoiled_broth.config import *
import warnings, os, logging, time, threading, csv, sys
import numpy as np
import cv2

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

training_map_nr = sys.argv[1]
num_agents = int(sys.argv[2])
intent_version = sys.argv[3]
cooperative = int(sys.argv[4])
game_version = sys.argv[5]
training_id = sys.argv[6]
checkpoint_number = sys.argv[7]
ENABLE_VIDEO_RECORDING = False if len(sys.argv) > 8 and str(sys.argv[8]).lower() == "false" else True

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
base_path = Path(f"/mnt/lustre/home/samuloza/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{training_map_nr}")
if cooperative:
    base_path = base_path / f"cooperative/Training_{training_id}"
else:
    base_path = base_path / f"competitive/Training_{training_id}"

path_root = Path().resolve() / "spoiled_broth"
checkpoint_dir = base_path / f"checkpoint_{checkpoint_number}"

# Determine grid size from map
map_txt_path = path_root / "maps" / f"{training_map_nr}.txt"
with open(map_txt_path) as f:
    map_lines = [line.rstrip("\n") for line in f.readlines()]
grid_size = (len(map_lines[0]), len(map_lines)) if map_lines else (8, 8)

# Initialize Ray
ray.shutdown()
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)

# Agent controllers
use_lstm = False
config_path = base_path / "config.txt"
if config_path.exists():
    with open(config_path) as f:
        for line in f:
            if line.strip().startswith("USE_LSTM"):
                use_lstm = line.strip().split(":")[1].strip().lower() == "true"
controller_cls = RLlibControllerLSTM if use_lstm else RLlibController

agent_map = {}
for i in range(1, num_agents + 1):
    agent_id = f"ai_rl_{i}"
    agent_map[agent_id] = controller_cls(agent_id, checkpoint_dir, f"policy_{agent_id}",
                                         competition=(game_version.upper() == "COMPETITION"))

# CSV logging
csv_filename = base_path / f"simulation_log_checkpoint_{checkpoint_number}_{timestamp}.csv"
state_logger = None
class GameStateLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Agent", "X", "Y", "Inventory", "Score"])
    def log_state(self, frame_idx, game):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for agent_id, agent in game.gameObjects.items():
                if agent_id.startswith("ai_rl_"):
                    writer.writerow([
                        frame_idx,
                        agent_id,
                        getattr(agent, "x", 0),
                        getattr(agent, "y", 0),
                        getattr(agent, "item", ""),
                        getattr(agent, "score", 0)
                    ])
state_logger = GameStateLogger(csv_filename)

# Video recorder
VIDEO_FPS = 24
class VideoRecorder:
    def __init__(self, output_path, fps=VIDEO_FPS):
        self.output_path = output_path
        self.fps = fps
        self.writer = None
    def start(self, frame_shape):
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w,h))
    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)
    def stop(self):
        if self.writer:
            self.writer.release()

recorder = None
renderer = Renderer2DOffline(path_root=path_root, tile_size=16)

# Game factory
def game_factory():
    game = Game(map_nr=training_map_nr, grid_size=grid_size, intent_version=intent_version)
    return game

# AI-only session
engine_app = AIOnlySessionApp(
    game_factory=game_factory,
    ui_modules=[],  # no Renderer2DModule
    agent_map=agent_map,
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=cf_AI_TICK_RATE,
    is_max_speed=True
)
engine_app.session_manager.start_session()
session = engine_app.get_session()
game = session.engine.game

# Simulation loop
FIXED_DURATION_SECONDS = 180
total_frames = FIXED_DURATION_SECONDS * VIDEO_FPS

def game_to_render_state(game):
    """Convert gameObjects to renderer-friendly dicts with 'id', 'x', 'y', 'type'."""
    state = []
    for obj_id, obj in game.gameObjects.items():
        state.append({
            "id": obj_id,
            "x": getattr(obj, "x", 0),
            "y": getattr(obj, "y", 0),
            "type": getattr(obj, "_type", "agent")
        })
    return {"gameObjects": state}

frame_idx = 0

# Video recorder initialization
if ENABLE_VIDEO_RECORDING:
    first_frame = renderer.render_to_array(None, game_to_render_state(game), 0)
    recorder = VideoRecorder(base_path / f"offline_recording_{timestamp}.mp4")
    recorder.start(first_frame.shape)

prev_state = None
progress_step = int(total_frames * 0.05)
next_progress = progress_step
for frame_idx in range(total_frames):
    t = (frame_idx % (VIDEO_FPS/24)) / (VIDEO_FPS/24)  # interpolation factor

    prev_state = prev_state or game_to_render_state(game)
    curr_state = game_to_render_state(game)

    frame = renderer.render_to_array(prev_state, curr_state, t)

    # Log agent states
    state_logger.log_state(frame_idx, game)

    # Video recording
    if ENABLE_VIDEO_RECORDING:
        recorder.write(frame)

    prev_state = curr_state

    # --- Advance game logic ---
    actions = {aid: None for aid in game.gameObjects if aid.startswith("ai_rl_")}
    game.step(actions, 1/VIDEO_FPS)  # reemplaza game.update({})

    # Print progress every 5%
    if frame_idx + 1 >= next_progress:
        percent = int(100 * (frame_idx + 1) / total_frames)
        print(f"Simulation progress: {percent}% ({frame_idx + 1}/{total_frames} frames)")
        next_progress += progress_step

# --- Finalizar ---
if recorder:
    recorder.stop()
engine_app.stop()
print("Simulation finished!")
print(f"CSV log: {csv_filename}")
if ENABLE_VIDEO_RECORDING:
    print(f"Video saved: {recorder.output_path}")
