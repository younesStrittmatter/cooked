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

training_map_nr = sys.argv[1]
num_agents = int(sys.argv[2])
intent_version = sys.argv[3]
cooperative = int(sys.argv[4])
game_version = sys.argv[5]  # classic, competition
training_id = sys.argv[6]
checkpoint_number = int(sys.argv[7])

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
FIXED_DURATION_SECONDS = 180  # 3 minutos exactos

# Configuración de grabación de vídeo
VIDEO_FPS = 60
VIDEO_SPEED_MULTIPLIER = 1.0
FRAME_SKIP = 1
ENABLE_VIDEO_RECORDING = True

print(f"Game Configuration:")
print(f"  Fixed Duration: {FIXED_DURATION_SECONDS} seconds")
print(f"  Video FPS: {VIDEO_FPS}")
print(f"  Video Speed Multiplier: {VIDEO_SPEED_MULTIPLIER}x")
print(f"  Frame Skip: {FRAME_SKIP}")
print(f"  Effective Video FPS: {VIDEO_FPS * VIDEO_SPEED_MULTIPLIER:.0f}")
print(f"  Actual Capture FPS: {VIDEO_FPS / FRAME_SKIP:.1f}")
print(f"  Video Recording: {'Enabled' if ENABLE_VIDEO_RECORDING else 'Disabled'}")
print()

#local = '/mnt/lustre/home/samuloza'
local = ''
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
            if scores is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (0, 255, 0)
                thickness = 2

                # Crear una barra negra lo suficientemente alta para n agentes
                line_height = 25
                bar_height = line_height * len(scores) + 20
                bar = np.zeros((bar_height, frame.shape[1], 3), dtype=np.uint8)

                # Escribir cada contador en una línea distinta
                for i, (agent, score) in enumerate(scores.items()):
                    text = f"{agent}: {score}"
                    y = (i + 1) * line_height
                    cv2.putText(bar, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

                # Apilar la barra encima del frame
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

        recorder = VideoRecorder(output_path=f"{base_path}/ai_only_recording_checkpoint_{checkpoint_number}.mp4", fps=fps)
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

                # Get scores from running engine
                session = engine_app.get_session()
                scores = session.engine.game.agent_scores

                recorder.capture_frame(frame, scores=scores)

                # Log progress once per second
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
engine_app = AIOnlySessionApp(
    game_factory=partial(Game, map_nr=playing_map_nr, grid_size=grid_size, intent_version=intent_version),
    ui_modules=[Renderer2DModule()],
    agent_map=agent_map,
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=cf_AI_TICK_RATE,
    is_max_speed=False
)

app = engine_app.app

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    def run_app():
        app.run(port=free_port, debug=False, use_reloader=False)
    
    app_thread = threading.Thread(target=run_app)
    app_thread.daemon = True
    app_thread.start()
    
    # Wait for the app to start
    time.sleep(3)
    
    if ENABLE_VIDEO_RECORDING:
        try:
            # Captura frames durante 180s
            capture_canvas_frames(
                url=f"http://localhost:{free_port}",
                duration_seconds=FIXED_DURATION_SECONDS,
                fps=VIDEO_FPS,
                frame_skip=FRAME_SKIP
            )
        except Exception as e:
            print(f"Video recording failed: {e}")
            print("Falling back to running game without video recording...")
            time.sleep(FIXED_DURATION_SECONDS)
    else:
        print("Video recording disabled. Running game for specified duration...")
        time.sleep(FIXED_DURATION_SECONDS)

    # Stop the session
    engine_app.stop()
    print("Game execution completed!")
