import ray
from engine.app.ai_only_app import AIOnlySessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
from spoiled_broth.rl.rllib_controller_lstm import RLlibControllerLSTM
from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.game_env_competition import GameEnvCompetition
from engine.extensions.renderer2d.renderer_2d_offline import Renderer2DOffline
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path
from spoiled_broth.config import *
import warnings, os, logging, time, threading, csv, sys
import numpy as np
import cv2

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# Cluster configuration
CLUSTER = 'cuenca'  # 'brigit', 'cuenca', or 'local'

# Simulation parameters
training_map_nr = sys.argv[1]
num_agents = int(sys.argv[2])
intent_version = sys.argv[3]
cooperative = int(sys.argv[4])
game_version = sys.argv[5]
training_id = sys.argv[6]
checkpoint_number = sys.argv[7]
ENABLE_VIDEO_RECORDING = False if len(sys.argv) > 8 and str(sys.argv[8]).lower() == "false" else True

timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")

if CLUSTER == 'brigit':
    local = '/mnt/lustre/home/samuloza'
elif CLUSTER == 'cuenca':
    local = ''
elif CLUSTER == 'local':
    local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
else:
    raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")

base_path = Path(f"{local}/data/samuel_lozano/cooked/{game_version}/{intent_version}/map_{training_map_nr}")

if cooperative:
    base_path = base_path / f"cooperative/Training_{training_id}"
else:
    base_path = base_path / f"competitive/Training_{training_id}"

path_root = Path().resolve() / "spoiled_broth"
checkpoint_dir = base_path / f"checkpoint_{checkpoint_number}"
saving_path = base_path / "simulations"
os.makedirs(saving_path, exist_ok=True)

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
    try:
        agent_map[agent_id] = controller_cls(agent_id, checkpoint_dir, f"policy_{agent_id}",
                                             competition=(game_version.upper() == "COMPETITION"))
    except Exception as e:
        print(f"Failed to initialize controller {agent_id}: {e}")

# CSV logging
csv_filename = saving_path / f"simulation_log_checkpoint_{checkpoint_number}_{timestamp}.csv"
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
renderer = None

# Game factory
def game_factory():
    game = Game(map_nr=training_map_nr, grid_size=grid_size, intent_version=intent_version)

    # Ensure initial state has no items on tiles or in agents' hands right after creation
    try:
        grid = getattr(game, 'grid', None)
        if grid is not None:
            for x in range(grid.width):
                for y in range(grid.height):
                    tile = grid.tiles[x][y]
                    if tile is None:
                        continue
                    if hasattr(tile, 'item'):
                        tile.item = None
                    if hasattr(tile, 'cut_time_accumulated'):
                        try:
                            tile.cut_time_accumulated = 0
                        except Exception:
                            pass
                    if hasattr(tile, 'cut_by'):
                        try:
                            tile.cut_by = None
                        except Exception:
                            pass
        # Also ensure any pre-existing agents have empty hands
        for aid, obj in list(game.gameObjects.items()):
            if aid.startswith('ai_rl_') and obj is not None:
                if hasattr(obj, 'item'):
                    obj.item = None
                if hasattr(obj, 'action_complete'):
                    obj.action_complete = True
                if hasattr(obj, 'current_action'):
                    obj.current_action = None
    except Exception as _e:
        print(f"Warning (game_factory): could not clear initial items: {_e}")

    return game

# AI-only session
engine_app = AIOnlySessionApp(
    game_factory=game_factory,
    ui_modules=[Renderer2DModule()],
    agent_map=agent_map,
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=cf_AI_TICK_RATE,
    is_max_speed=False
)
session = engine_app.get_session()
game = session.engine.game

# Ensure initial state has no items on tiles or in agents' hands
try:
    grid = getattr(game, 'grid', None)
    if grid is not None:
        for x in range(grid.width):
            for y in range(grid.height):
                tile = grid.tiles[x][y]
                if tile is None:
                    continue
                # Clear any placed item on counters/cutting boards
                if hasattr(tile, 'item'):
                    tile.item = None
                # Reset cutting progress if present
                if hasattr(tile, 'cut_stage'):
                    try:
                        tile.cut_stage = 0
                    except Exception:
                        pass
    # Clear any items agents might be holding and reset action state
    for aid, obj in list(game.gameObjects.items()):
        if aid.startswith('ai_rl_') and obj is not None:
            if hasattr(obj, 'item'):
                obj.item = None
            if hasattr(obj, 'action_complete'):
                obj.action_complete = True
            if hasattr(obj, 'current_action'):
                obj.current_action = None
except Exception as _e:
    print(f"Warning: could not clear initial items: {_e}")

# Debug: print runner and agent thread status
runner = getattr(session, '_runner', None)
if runner:
    print(f"EngineRunner created: tick_rate={runner.tick_rate}, ai_tick_rate={runner.ai_tick_rate}, is_max_speed={runner.is_max_speed}")
    print(f"Engine thread alive: {runner.engine_thread.is_alive()}")
    if runner.agent_thread:
        print(f"Agent thread alive: {runner.agent_thread.is_alive()}")
    else:
        print("No agent thread attached (agent_map may be empty)")

# Simulation loop
FIXED_DURATION_SECONDS = 180
total_frames = FIXED_DURATION_SECONDS * VIDEO_FPS

def serialize_ui_state(game, engine, agent_id=None):
    """Serialize payload using UI modules (same output as /state route).

    This uses the session.ui_modules (CoreUIModule + Renderer2DModule) to produce
    the same `gameObjects` list the browser uses, so offline rendering matches online.
    """
    payload = {}
    for module in session.ui_modules:
        try:
            payload.update(module.serialize_for_agent(game, engine, agent_id))
        except Exception as e:
            print(f"UI module serialization failed: {e}")
    payload["tick"] = engine.tick_count
    return payload

frame_idx = 0

# Video recorder will be initialized lazily once we produce the first frame

prev_state = None
progress_step = int(total_frames * 0.05)
next_progress = progress_step
for frame_idx in range(total_frames):
    t = (frame_idx % (VIDEO_FPS/24)) / (VIDEO_FPS/24)  # interpolation factor

    # Serialize UI state via the installed UI modules
    curr_state = serialize_ui_state(game, session.engine)
    prev_state = prev_state or curr_state

    if renderer is None:
        renderer = Renderer2DOffline(path_root=path_root, tile_size=16)
        # initialize writer with a first generated frame
        first_frame = renderer.render_to_array(prev_state, curr_state, 0.0)
        if ENABLE_VIDEO_RECORDING:
            recorder = VideoRecorder(saving_path / f"offline_recording_{timestamp}.mp4")
            recorder.start(first_frame.shape)

    frame = renderer.render_to_array(prev_state, curr_state, 0.0)

    # Log agent states
    state_logger.log_state(frame_idx, game)

    if ENABLE_VIDEO_RECORDING and recorder:
        recorder.write(frame)

    prev_state = curr_state

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
