from engine.app.ai_only_app import AIOnlySessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.rllib_controller import RLlibController
from spoiled_broth.rl.rllib_controller_lstm import RLlibControllerLSTM
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path
from spoiled_broth.config import *
from functools import partial
from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.game_env_competition import GameEnvCompetition
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
import time
import threading
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
from io import BytesIO
from PIL import Image
from flask import jsonify
import signal
import sys
import socket
import csv
from datetime import datetime

training_map_nr = sys.argv[1]
num_agents = int(sys.argv[2])
intent_version = sys.argv[3]
cooperative = int(sys.argv[4])
game_version = sys.argv[5]  # classic, competition
training_id = sys.argv[6]
checkpoint_number = sys.argv[7]
# Optional parameter for video recording, defaults to True if not provided
ENABLE_VIDEO_RECORDING = False if len(sys.argv) > 8 and str(sys.argv[8]).lower() == "false" else True
# Setup CSV logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

playing_map_nr = training_map_nr

def find_free_port(start_port=5000, max_tries=100):
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                s.close()
                return port
            except OSError:
                port += 1
    raise RuntimeError("Could not find a free port in range.")

# Find first free port starting from 5000
free_port = find_free_port(start_port=5000)
print(f"Using free port: {free_port}")

# --- Determine grid size from map file ---
map_txt_path = Path(__file__).parent / 'spoiled_broth' / 'maps' / f'{training_map_nr}.txt'
with open(map_txt_path, 'r') as f:
    map_lines = [line.rstrip('\n') for line in f.readlines()]
rows = len(map_lines)
cols = len(map_lines[0]) if rows > 0 else 0
grid_size = (cols, rows)  # (width, height)

# Configuration for game duration and actions
FIXED_DURATION_SECONDS = 180  # 3 minutes

# Configuración de grabación de vídeo
VIDEO_FPS = 60
VIDEO_SPEED_MULTIPLIER = 1.0
FRAME_SKIP = 1

print(f"Game Configuration:")
print(f"  Fixed Duration: {FIXED_DURATION_SECONDS} seconds")
print(f"  Video FPS: {VIDEO_FPS}")
print(f"  Video Speed Multiplier: {VIDEO_SPEED_MULTIPLIER}x")
print(f"  Frame Skip: {FRAME_SKIP}")
print(f"  Effective Video FPS: {VIDEO_FPS * VIDEO_SPEED_MULTIPLIER:.0f}")
print(f"  Actual Capture FPS: {VIDEO_FPS / FRAME_SKIP:.1f}")
print(f"  Video Recording: {'Enabled' if ENABLE_VIDEO_RECORDING else 'Disabled'}")
print()

local = '/mnt/lustre/home/samuloza'
#local = ''
#local = 'C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'

if num_agents == 1:
    raw_dir = f'{local}/data/samuel_lozano/cooked/pretraining'
else:
    raw_dir = f'{local}/data/samuel_lozano/cooked'

if intent_version is not None:
    intent_dir = f'{raw_dir}/{game_version}/{intent_version}'
else:
    intent_dir = f'{raw_dir}/{game_version}'

if cooperative: 
    base_path = Path(f"{intent_dir}/map_{training_map_nr}/cooperative/Training_{training_id}")
else:
    base_path = Path(f"{intent_dir}/map_{training_map_nr}/competitive/Training_{training_id}")

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

csv_filename = f"{base_path}/simulation_log_checkpoint_{checkpoint_number}_{timestamp}.csv"

def env_creator(config):
    return GameEnv(**config)

def env_creator_competition(config):
    return GameEnvCompetition(**config)

# Initialize Ray manually BEFORE RLlib does it automatically
ray.shutdown()
ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)

if game_version.upper() == "CLASSIC":
    register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator(config)))
elif game_version.upper() == "COMPETITION":
    register_env("spoiled_broth", lambda config: ParallelPettingZooEnv(env_creator_competition(config)))

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path().resolve() / "spoiled_broth"
checkpoint_dir = Path(f"{base_path}/checkpoint_{checkpoint_number}/")
if not checkpoint_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

class VideoRecorder:
    def __init__(self, output_path="game_recording.mp4", fps=24):
        self.output_path = output_path
        self.fps = fps
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        
    def start_recording(self):
        self.recording = True
        self.frame_count = 0
        print(f"Started recording video to {self.output_path}")
        
    def capture_frame(self, frame, scores=None):
        if self.recording:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (0, 255, 0)
            thickness = 2

            # Calculate total height needed for all information
            line_height = 25
            info_lines = 1  # Start with 1 for the scores header
            if scores:
                info_lines += len(scores)

            bar_height = line_height * info_lines + 20
            bar = np.zeros((bar_height, frame.shape[1], 3), dtype=np.uint8)

            current_y = line_height

            # Draw scores
            if scores:
                for i, (agent, score) in enumerate(scores.items()):
                    text = f"{agent}: {score}"
                    current_y += line_height
                    cv2.putText(bar, text, (10, current_y), font, font_scale, color, thickness, cv2.LINE_AA)

            # Stack the info bar on top of the frame
            frame = np.vstack((bar, frame))

            # Initialize writer if needed
            if self.video_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_path, fourcc, int(self.fps * VIDEO_SPEED_MULTIPLIER), (width, height))

            self.video_writer.write(frame)
            self.frame_count += 1
            
    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved successfully to {self.output_path} ({self.frame_count} frames)")
            
    def save_video(self):
        # This method is no longer needed as frames are written directly
        pass

class ActionTracker:
    def __init__(self, csv_path):
        self.csv_path = csv_path.replace('.csv', '_actions.csv')
        self.current_actions = {}  # track ongoing actions by agent
        self.start_frame = time.time()  # Reference time for frame counting
        
        # Create CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Agent',
                'Action_Number',  # The number from the action space
                'Tile_Type',  # The type of tile clicked (e.g., counter, cutting_board, etc.)
                'Start_Frame',  # Frame when action started
                'End_Frame',  # Frame when action ended
                'Start_Position_X',  # Agent's starting position
                'Start_Position_Y',
                'Target_Position_X',  # Clicked tile position
                'Target_Position_Y',
                'Start_Item',  # What the agent was holding at start
                'End_Item',  # What the agent was holding at end
                'Start_Time',
                'End_Time',
                'Duration'
            ])

    def start_action(self, agent_id, action_type, target_tile, timestamp, action_number=None):
        # Only start a new action if there isn't one in progress
        if agent_id not in self.current_actions:
            # Handle both string ID and tile object cases
            if target_tile is None:
                game = next(iter(self.current_actions.values()))['game'] if self.current_actions else None
                if not game:
                    return
                agent = game.gameObjects[agent_id]
                tile_type = "unknown"
                target_x = -1
                target_y = -1
            else:
                game = target_tile.game
                agent = game.gameObjects[agent_id]
                tile_type = type(target_tile).__name__.lower()
                target_x = target_tile.slot_x
                target_y = target_tile.slot_y
            
            # Get tile type name (e.g., "counter", "cutting_board", etc.)
            tile_type = type(target_tile).__name__.lower()
            
            # Get current item the agent is holding
            start_item = getattr(agent, "item", None)
            start_item = str(start_item) if start_item is not None else "None"
            
            # Calculate current frame
            current_frame = int((timestamp - self.start_frame) * cf_AI_TICK_RATE)
            
            self.current_actions[agent_id] = {
                'game': game,
                'action_number': action_number if action_number is not None else -1,  # -1 for unknown
                'tile_type': tile_type,
                'start_frame': current_frame,
                'start_pos_x': agent.slot_x,
                'start_pos_y': agent.slot_y,
                'target_pos_x': target_x,
                'target_pos_y': target_y,
                'start_item': start_item,
                'start_time': timestamp
            }

    def end_action(self, agent_id, timestamp, game=None):
        if agent_id in self.current_actions:
            action = self.current_actions[agent_id]
            duration = timestamp - action['start_time']

            # Get the ending frame
            end_frame = int((timestamp - self.start_frame) * cf_AI_TICK_RATE)

            # Use the latest game state to get the agent's current item
            agent = None
            if game is not None:
                agent = game.gameObjects.get(agent_id, None)
            else:
                # Fallback to previous method if game not provided
                for obj in next(iter(self.current_actions.values()))['game'].gameObjects.values():
                    if hasattr(obj, 'id') and obj.id == agent_id:
                        agent = obj
                        break

            end_item = "None"
            if agent:
                end_item = getattr(agent, "item", None)
                end_item = str(end_item) if end_item is not None else "None"

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    agent_id,
                    action['action_number'],
                    action['tile_type'],
                    action['start_frame'],
                    end_frame,
                    action['start_pos_x'],
                    action['start_pos_y'],
                    action['target_pos_x'],
                    action['target_pos_y'],
                    action['start_item'],
                    end_item,
                    action['start_time'],
                    timestamp,
                    f"{duration:.3f}"
                ])

            del self.current_actions[agent_id]

    def has_action_in_progress(self, agent_id):
        return agent_id in self.current_actions

class GameStateLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        # Create CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp', 'Agent', 'Position_X', 'Position_Y', 
                           'Inventory', 'Score'])
    
    def log_state(self, frame_idx, game):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = time.time()
            
            for agent_id, agent in game.gameObjects.items():
                if agent_id.startswith('ai_rl_'):
                    # Get grid position
                    grid_x = agent.x // game.grid.tile_size
                    grid_y = agent.y // game.grid.tile_size
                    
                    # Get agent's inventory and state
                    inventory = agent.item if hasattr(agent, 'item') else []
                    
                    # Write row to CSV
                    writer.writerow([
                        frame_idx,
                        timestamp,
                        agent_id,
                        f"{grid_x:.1f}",
                        f"{grid_y:.1f}",
                        str(inventory),
                        agent.score
                    ])

def capture_canvas_frames(
    url="http://localhost:5000",
    duration_seconds=180,
    fps=60,
    frame_skip=1,
    video_speed_multiplier=1.0
):
    """Capture frames from the game canvas using Selenium for a fixed duration (real time)."""

    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # modern headless
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,720")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        time.sleep(2)  # wait for the page to load

        recorder = VideoRecorder(output_path=f"{base_path}/ai_only_recording_checkpoint_{checkpoint_number}_{timestamp}.mp4", fps=fps)
        recorder.start_recording()

        total_frames = int(duration_seconds * fps)
        start_time = time.time()

        for frame_idx in range(total_frames):
            try:
                # Grab canvas -> base64
                canvas_base64 = driver.execute_script("""
                    var canvas = document.getElementById('scene');
                    if (!canvas) { throw new Error('Canvas not found'); }
                    return canvas.toDataURL('image/png').substring(22);
                """)
                img_data = base64.b64decode(canvas_base64)
                img = Image.open(BytesIO(img_data))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # Get game state information
                session = engine_app.get_session()
                game = session.engine.game
            
                # Log state to CSV
                state_logger.log_state(frame_idx, game)

                # Collect agent information for video if recording is enabled
                if ENABLE_VIDEO_RECORDING:
                    scores = game.agent_scores
                    recorder.capture_frame(frame, scores=scores) 

                if frame_idx % fps == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({elapsed:.1f}s elapsed)")

                # Keep real-time pacing
                target_time = start_time + (frame_idx + 1) / fps
                sleep_time = target_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"Error capturing frame {frame_idx}: {e}")
                continue

        recorder.stop_recording()

    finally:
        if driver:
            driver.quit()

use_lstm = False
if config_path.exists():
    with open(config_path, "r") as f:
        for line in f:
            if line.strip().startswith("USE_LSTM"):
                key, value = line.strip().split(":", 1)
                value = value.strip().lower()
                if value == "true":
                    use_lstm = True
                break
print(f"\nUsing LSTM controller: {'Yes' if use_lstm else 'No'}\n")

# Choose controller class
controller_cls = RLlibControllerLSTM if use_lstm else RLlibController

# Create the agent_map dynamically
if num_agents == 1:
    agent_map = {
        "ai_rl_1": controller_cls("ai_rl_1", checkpoint_dir, "policy_ai_rl_1", competition=(game_version.upper() == "COMPETITION"))
    }
else:
    agent_map = {
        "ai_rl_1": controller_cls("ai_rl_1", checkpoint_dir, "policy_ai_rl_1", competition=(game_version.upper() == "COMPETITION")),
        "ai_rl_2": controller_cls("ai_rl_2", checkpoint_dir, "policy_ai_rl_2", competition=(game_version.upper() == "COMPETITION"))
    }

# Create the AI-only engine app
if __name__ == "__main__":
    # Initialize recording setup
    driver = None
    recorder = None
    
    # Setup CSV logging and tracking first
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    csv_filename = f"{base_path}/simulation_log_checkpoint_{checkpoint_number}_{timestamp}.csv"
    state_logger = GameStateLogger(csv_filename)
    action_tracker = ActionTracker(csv_filename)

    def game_factory():
        game = Game(map_nr=playing_map_nr, grid_size=grid_size, intent_version=intent_version)
        # Add action tracker to the game instance so controllers can access it
        game.action_tracker = action_tracker
        
        # Initialize action completion states for all agents
        for agent_id, agent in game.gameObjects.items():
            if hasattr(agent, 'is_agent') and agent.is_agent:
                agent.action_complete = True
                agent.current_action = None
        
        return game

    engine_app = AIOnlySessionApp(
        game_factory=game_factory,
        ui_modules=[Renderer2DModule()],
        agent_map=agent_map,
        path_root=path_root,
        tick_rate=24,
        ai_tick_rate=cf_AI_TICK_RATE,
        is_max_speed=False
    )

    app = engine_app.app

    print(f"Logging simulation data to: {csv_filename}")
    print(f"Logging action data to: {csv_filename.replace('.csv', '_actions.csv')}")
    
    # Start the Flask app in a separate thread
    def run_app():
        app.run(port=free_port, debug=False, use_reloader=False)
    
    app_thread = threading.Thread(target=run_app)
    app_thread.daemon = True
    app_thread.start()
    
    # Wait for the Flask app to start
    time.sleep(2)
    
    # Initialize video recording if enabled
    if ENABLE_VIDEO_RECORDING:
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1280,720")
            driver = webdriver.Chrome(options=chrome_options)
            
            driver.get(f"http://localhost:{free_port}")
            time.sleep(1)  # Wait for page to load
            
            recorder = VideoRecorder(
                output_path=f"{base_path}/ai_only_recording_checkpoint_{checkpoint_number}_{timestamp}.mp4",
                fps=VIDEO_FPS
            )
            recorder.start_recording()
            print(f"Video recording started: {recorder.output_path}")
        except Exception as e:
            print(f"Failed to initialize video recording: {e}")
            if driver:
                driver.quit()
            driver = None
            recorder = None
    
    # Start the game session
    print("Starting game simulation...")
    engine_app.session_manager.start_session()
    
    # Main simulation loop
    start_time = time.time()
    try:
        for second in range(FIXED_DURATION_SECONDS):
            # Get game state
            session = engine_app.get_session()
            game = session.engine.game
            
            # Log state to CSV (once per second)
            state_logger.log_state(second, game)
            
            # Handle video recording if enabled
            if ENABLE_VIDEO_RECORDING and driver and recorder:
                try:
                    # Capture frames for this second
                    for frame in range(VIDEO_FPS):
                        frame_idx = second * VIDEO_FPS + frame
                        
                        # Grab canvas
                        canvas_base64 = driver.execute_script("""
                            var canvas = document.getElementById('scene');
                            if (!canvas) { throw new Error('Canvas not found'); }
                            return canvas.toDataURL('image/png').substring(22);
                        """)
                        img_data = base64.b64decode(canvas_base64)
                        img = Image.open(BytesIO(img_data))
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        
                        recorder.capture_frame(frame, scores=game.agent_scores)
                        
                        # Maintain frame timing
                        frame_target = start_time + (frame_idx + 1) / VIDEO_FPS
                        sleep_time = frame_target - time.time()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                except Exception as e:
                    print(f"Error in video recording at second {second}: {e}")
            
            else:
                # If not recording video, just wait one second
                next_second = start_time + (second + 1)
                sleep_time = next_second - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Print progress
            if second % 10 == 0:  # Every 10 seconds
                progress = (second / FIXED_DURATION_SECONDS) * 100
                elapsed = time.time() - start_time
                print(f"Simulation progress: {progress:.1f}% ({elapsed:.1f}s elapsed)")
    
    finally:
        # Cleanup
        if recorder:
            recorder.stop_recording()
        if driver:
            driver.quit()
        engine_app.stop()
        print("Simulation completed!")
        print(f"Data logged to: {csv_filename}")
        if ENABLE_VIDEO_RECORDING and recorder:
            print(f"Video saved to: {recorder.output_path}")

    # Stop the session
    engine_app.stop()
    print("Game execution completed!")
